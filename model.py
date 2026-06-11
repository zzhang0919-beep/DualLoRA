import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from util import act

class GNN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn_type='TransformerConv', gnn_layer_num=2):
        super().__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([GraphConv(input_dim, 2 * out_dim), GraphConv(2 * out_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim)]
            for i in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim))
            layers.append(GraphConv(2 * out_dim, out_dim))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.conv[0:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
        node_emb = self.conv[-1](x, edge_index)
        return node_emb

class GATConv_lora(nn.Module):
    def __init__(self, in_channels, out_channels, r=4, lora_alpha=8, heads=1, concat=False):
        super(GATConv_lora, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.r = r
        self.scaling = lora_alpha / r

        self.lin_feat_b = nn.Linear(in_channels, r, bias=False)   
        self.lin_feat_a = nn.Linear(r, out_channels, bias=False)  

        self.lin_struct_b = nn.Linear(in_channels, r, bias=False) 
        self.lin_struct_a = nn.Linear(r, out_channels, bias=False)
        
        self.msg_passer = GATConv(
            in_channels=out_channels, 
            out_channels=out_channels, 
            heads=heads, 
            concat=concat,
            bias=False
        )

        self.current_z_feat = None
        self.current_z_struct = None
        
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lin_feat_b.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lin_feat_a.weight)
        
        nn.init.kaiming_uniform_(self.lin_struct_b.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lin_struct_a.weight)
        
        if hasattr(self.msg_passer, 'reset_parameters'):
            self.msg_passer.reset_parameters()

    def forward(self, x, edge_index):
        z_feat = self.lin_feat_b(x)      
        h_feat = self.lin_feat_a(z_feat) 
        
        self.current_z_feat = z_feat 

        z_struct = self.lin_struct_b(x)   

        self.current_z_struct = z_struct
        
        z_struct_recon = self.lin_struct_a(z_struct)      
        h_struct = self.msg_passer(z_struct_recon, edge_index) 
        return self.scaling * (h_feat + h_struct)


class CME_GNN(nn.Module):
    def __init__(self, pretrain_dataset, pretrained_grace_model, config, input_dim=16, out_dim=32, r=4, gnn_layer_num=2, activation='prelu', args=None):
        super(CME_GNN, self).__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation_type = activation
        device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
        
        self.frozen_backbone = GNN(pretrain_dataset.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
        model_path = "./pre_trained_gnn/{}.{}.{}.{}.pth".format(args.pretrain_dataset, args.pretext, config['gnn_type'], args.is_reduction)
        self.frozen_backbone.load_state_dict(torch.load(model_path))
        self.frozen_backbone.to(device)
        self.frozen_backbone.eval()
        for param in self.frozen_backbone.conv.parameters():
            param.requires_grad = False
            
        with torch.no_grad():
            x_test = torch.zeros((2, pretrain_dataset.x.shape[1]), device=device)
            edge_test = torch.tensor([[0, 1], [1, 0]], device=device)
            backbone_output_dims = []
            h_test = x_test
            for backbone_layer in self.frozen_backbone.conv:
                h_test = backbone_layer(h_test, edge_test)
                backbone_output_dims.append(h_test.size(1))
                
        self.conv = nn.ModuleList()
        current_in_dim = input_dim 
        
        for idx in range(self.gnn_layer_num):
            target_out_dim = backbone_output_dims[idx] 
            self.conv.append(GATConv_lora(in_channels=current_in_dim, out_channels=target_out_dim, r=r))
            current_in_dim = target_out_dim

        self.node_emb1 = None

    def _get_activation(self, x):
        if self.activation_type == 'prelu':
            return F.prelu(x, torch.tensor(0.25, device=x.device))
        elif self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'elu':
            return F.elu(x)
        return x

    def forward(self, x, edge_index):
        h_current = x    
        h_teacher = x     
        
        backbone_layers = self.frozen_backbone.conv
        
        for i in range(self.gnn_layer_num):

            h_teacher = backbone_layers[i](h_teacher, edge_index)
            
            h_base = backbone_layers[i](h_current, edge_index)
            h_lora = self.conv[i](h_current, edge_index)
            h_combined = h_base + h_lora
            
            if i < self.gnn_layer_num - 1:
                h_teacher = self._get_activation(h_teacher)
                h_combined = self._get_activation(h_combined)
                h_current = h_combined  
                
        return h_combined, h_teacher, h_combined

class GNNLoRA(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn, gnn_type='GAT', gnn_layer_num=2, r=32):
        super().__init__()
        self.gnn = gnn
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv_lora
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.conv = nn.ModuleList()

        for i in range(len(self.gnn.conv)):
            base_layer = self.gnn.conv[i]
            
            in_c = base_layer.in_channels
            out_c = base_layer.out_channels
            heads = getattr(base_layer, 'heads', 1)
            concat = getattr(base_layer, 'concat', False)
            
            if gnn_type in ['GAT', 'TransformerConv']:
                self.conv.append(GraphConv(in_c, out_c, heads=heads, concat=concat, r=r))
            else:
                self.conv.append(GraphConv(in_c, out_c, r=r))

    def forward(self, x, edge_index):
        base_x = x
        
        for i in range(len(self.conv) - 1):

            base_out = self.gnn.conv[i](base_x, edge_index)

            lora_out = self.conv[i](x, edge_index)
            
            base_x = base_out + lora_out
            
            base_x = self.activation(base_x)
            x = base_x

        node_emb_base = self.gnn.conv[-1](x, edge_index)
        node_emb_lora = self.conv[-1](x, edge_index)
        
        return node_emb_base + node_emb_lora, node_emb_base, node_emb_lora
        

class ZeroMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ZeroMLP, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.zeros_(self.linear.weight)  

    def forward(self, x):
        return self.linear(x)

class GraphControlGNN(nn.Module):
    def __init__(self, pretrain_dataset, pretrained_grace_model, config, input_dim=500, out_dim=32, r=16, gnn_layer_num=2):
        super(GraphControlGNN, self).__init__()
        
        self.frozen_backbone = nn.ModuleList([
            GATConv(input_dim, out_dim, bias=False),
            GATConv(out_dim, out_dim, bias=False)
        ])
        
        self.trainable_copy = nn.ModuleList([
            GATConv(input_dim, out_dim, bias=False),
            GATConv(out_dim, out_dim, bias=False)
        ])
        
        if pretrained_grace_model and os.path.exists(pretrained_grace_model):
            try:
                ckpt = torch.load(pretrained_grace_model, map_location='cpu')
                self.frozen_backbone.load_state_dict(ckpt, strict=False)
                self.trainable_copy.load_state_dict(ckpt, strict=False)
            except Exception as e:
                print(f"-> [GraphControl]{e}")
                
        for param in self.frozen_backbone.parameters():
            param.requires_grad = False
            
        self.Z1 = ZeroMLP(in_dim=r, out_dim=input_dim)  
        self.Z2 = ZeroMLP(in_dim=out_dim, out_dim=out_dim) 

    def forward(self, x, edge_index, P_prime=None):
        h_frozen = x
        for layer in self.frozen_backbone:
            h_frozen = layer(h_frozen, edge_index)
            
        if P_prime is not None:
            x_controlled = x + self.Z1(P_prime)
        else:
            x_controlled = x
            
        h_trainable = x_controlled
        for layer in self.trainable_copy:
            h_trainable = layer(h_trainable, edge_index)
            
        emb = h_frozen + self.Z2(h_trainable)
        return emb, emb, emb