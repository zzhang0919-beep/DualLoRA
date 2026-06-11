import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
import copy
import argparse
import numpy as np
import yaml
import os
import random
from yaml import SafeLoader
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import SVDFeatureReduction
from model import CME_GNN, GNN
from pre_train import pretrain
from memory_bank import MemoryBank
from interceptor import GradientInterceptor
from train import train_single_task_cme
from baseline import train_single_task_baseline   
from util import get_dataset, act, SMMDLoss, mkdir, get_parameter, Projector, LogReg, batched_smmd_loss, batched_gct_loss, get_ppr_weight

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PretrainedGRACE(nn.Module):
    def __init__(self, gnn_backbone):
        super().__init__()
        self.gnn = gnn_backbone 

def run_continual_experiment():
    parser = argparse.ArgumentParser(description='HetLoRA')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_tasks', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--alpha_base', type=float, default=0.8)
    parser.add_argument('--gamma_topo', type=float, default=1.0)
    parser.add_argument('--svd_threshold', type=float, default=1e-4)
    parser.add_argument('--lr_lora', type=float, default=0.001)
    parser.add_argument('--lr_head', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    parser.add_argument('--pretrain_dataset', type=str, default='PubMed')
    parser.add_argument('--is_pretrain', type=bool, default=False)
    parser.add_argument('--is_reduction', type=bool, default=True)
    parser.add_argument('--test_dataset', type=str, default='CiteSeer')
    parser.add_argument('--pretext', type=str, default='GRACE')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--few', type=bool, default=False)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--sup_weight', type=float, default=0.2)
    parser.add_argument('--method', type=str, default='cme', choices=['cme', 'baseline', 'calora','graphcontrol'])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--control_threshold', type=float, default=0.17)
    args = parser.parse_args()
    args = get_parameter(args)
   
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    dataset_path = f'./datasets/{args.dataset}'
    global_dataset = get_dataset(dataset_path, args.dataset)[0]

    global_dataset.edge_index = add_remaining_self_loops(global_dataset.edge_index)[0] 

    global_dataset = SVDFeatureReduction(out_channels=500)(global_dataset)  

    num_nodes = global_dataset.x.shape[0]
    input_dim = global_dataset.x.shape[1]
    
    total_classes = int(max(global_dataset.y).item() + 1)
    classes_per_task = total_classes // args.num_tasks
    
    print(f"{args.dataset} | {num_nodes} | {input_dim} | {total_classes}")

    memory_bank = MemoryBank(r=args.lora_r).to(device)
    interceptor = GradientInterceptor(alpha_base=args.alpha_base, gamma_topo=args.gamma_topo)

    if args.is_pretrain:
        config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)[args.pretrain_dataset]
        pretrain(args.pretrain_dataset, args.pretext, config_pretrain, args.gpu_id, args.is_reduction)
    grace_model = "./pre_trained_gnn/{}.{}.{}.{}.pth".format(args.pretrain_dataset, args.pretext, "GAT", args.is_reduction)
    
    pretrain_datapath = os.path.join('./datasets', args.pretrain_dataset)

    pretrain_dataset = get_dataset(pretrain_datapath, args.pretrain_dataset)[0]
    
    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]

    pretrain_dataset = SVDFeatureReduction(out_channels=500)(pretrain_dataset)
    pretrain_dataset = pretrain_dataset.to(device)

    test_dataset = global_dataset  

    config_transfer = yaml.load(open(args.config), Loader=SafeLoader)['transfer']

    if args.method == 'graphcontrol':
        from model import GraphControlGNN
        model = GraphControlGNN(
            pretrain_dataset=pretrain_dataset,
            pretrained_grace_model=grace_model,
            config=config_transfer,
            input_dim=500, 
            out_dim=32,
            r=args.lora_r,
            gnn_layer_num=2
        ).to(device)
    else:
        model = CME_GNN(
        pretrain_dataset=pretrain_dataset,
        pretrained_grace_model=grace_model,
        config=config_transfer,
        input_dim=input_dim,
        out_dim=32,
        r=args.lora_r,
        gnn_layer_num=2,
        activation='elu',
        args=args
    ).to(device)

    projector = Projector(input_dim, 500).to(device) 

    R_matrix = np.zeros((args.num_tasks, args.num_tasks))
    classifier_heads_store = {}
    global_dataset = global_dataset.to(device)
    
    target_adj = to_dense_adj(global_dataset.edge_index, max_num_nodes=num_nodes)[0].to(device)

    SMMD = SMMDLoss().to(device)
    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)
    ppr_weight = get_ppr_weight(test_dataset)
    mask = 1.0 - torch.eye(num_nodes, device=device)  

    if args.method == 'cme':
        P_prime = None
    elif args.method == 'cme' or args.method == 'baseline':
        P_prime = None
    else:
        with torch.no_grad():
            x_norm = F.normalize(global_dataset.x, p=2, dim=1)
            K = torch.mm(x_norm, x_norm.t())  
            A_prime = (K > args.control_threshold).float() 
            A_prime.fill_diagonal_(0)
            
            D_prime_vec = torch.sum(A_prime, dim=1)
            D_prime_vec = torch.where(D_prime_vec == 0, torch.ones_like(D_prime_vec), D_prime_vec)
            D_inv_sqrt = torch.diag(torch.pow(D_prime_vec, -0.5))
            L_prime = torch.eye(num_nodes, device=device) - torch.mm(torch.mm(D_inv_sqrt, A_prime), D_inv_sqrt)
            
            _, eigenvectors = torch.linalg.eigh(L_prime)
            P_prime = eigenvectors[:, :args.lora_r].to(device)

    for task_id in range(args.num_tasks):       
        current_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        
        task_node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        for cls in current_classes:
            task_node_mask = task_node_mask | (global_dataset.y == cls)
            
        task_nodes_idx = torch.nonzero(task_node_mask, as_tuple=False).squeeze().cpu().numpy()
        np.random.shuffle(task_nodes_idx)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        train_mask[task_nodes_idx[:args.shot]] = True
        val_mask[task_nodes_idx[args.shot:args.shot+10]] = True
        test_mask[task_nodes_idx[args.shot+20:]] = True
        
        with torch.no_grad():
            dummy_x = global_dataset.x[:2].to(device)
            
            dummy_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
            
            mapped_dummy_x = projector(dummy_x) 

            dummy_emb, _, _ = model(mapped_dummy_x, dummy_edge)
            actual_emb_dim = dummy_emb.shape[1]  
            
        logreg = LogReg(actual_emb_dim, total_classes).to(device)
        
        if args.method == 'cme':
            train_single_task_cme(
                task_id=task_id, 
                gnn2=model, 
                projector=projector, 
                logreg=logreg, 
                # test_dataset=test_dataset, 
                test_dataset=global_dataset,
                train_mask=train_mask, 
                target_adj=target_adj, 
                args=args, 
                device=device, 
                memory_bank=memory_bank, 
                interceptor=interceptor,
                batched_smmd_loss=batched_smmd_loss, 
                batched_gct_loss=batched_gct_loss,
                pretrain_graph_loader = pretrain_graph_loader,          
                SMMD=SMMD,                
                ppr_weight = ppr_weight,
                mask = mask
            )
        elif args.method == 'baseline':
            train_single_task_baseline(
                model=model, projector=projector, logreg=logreg,
                global_dataset=global_dataset, train_mask=train_mask,
                pretrain_graph_loader=pretrain_graph_loader, SMMD=SMMD, ppr_weight=ppr_weight, mask=mask,
                target_adj=target_adj,
                device=device, args=args
            )
        elif args.method == 'graphcontrol':
            from train import train_single_task_graphcontrol
            train_single_task_graphcontrol(
                task_id=task_id, gnn2=model, projector=projector, logreg=logreg,
                test_dataset=global_dataset, train_mask=train_mask, target_adj=target_adj,
                args=args, device=device, P_prime=P_prime, # 核心注入
                batched_smmd_loss=batched_smmd_loss, batched_gct_loss=batched_gct_loss,
                pretrain_graph_loader=pretrain_graph_loader, SMMD=SMMD, ppr_weight=ppr_weight, mask=mask
            )
        
        classifier_heads_store[task_id] = copy.deepcopy(logreg)

        model.eval()
        projector.eval()
        
        with torch.no_grad():
            mapped_x = projector(global_dataset.x)
            if args.method == 'graphcontrol':
                emb_eval, _, _ = model(mapped_x, global_dataset.edge_index, P_prime=P_prime)
            else:
                emb_eval, _, _ = model(mapped_x, global_dataset.edge_index)
            
            for past_task_id in range(task_id + 1):
                past_classes = list(range(past_task_id * classes_per_task, (past_task_id + 1) * classes_per_task))
                past_node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                for cls in past_classes:
                    past_node_mask = past_node_mask | (global_dataset.y == cls)
                
                head_eval = classifier_heads_store[past_task_id]
                head_eval.eval()

                past_logits = head_eval(emb_eval[past_node_mask])
                past_labels = global_dataset.y[past_node_mask]
                
                past_preds = torch.argmax(past_logits, dim=1)
                past_acc = (torch.sum(past_preds == past_labels).float() / past_labels.shape[0]).item()
                
                R_matrix[task_id, past_task_id] = past_acc

    print("R-Matrix:")
    print(np.round(R_matrix * 100, 2))
    
    final_task_id = args.num_tasks - 1
    average_performance = np.mean(R_matrix[final_task_id, :]) 
    
    bwt_list = []
    for k in range(args.num_tasks - 1):
        bwt_k = R_matrix[final_task_id, k] - R_matrix[k, k] 
        bwt_list.append(bwt_k)
    backward_transfer = np.mean(bwt_list) if len(bwt_list) > 0 else 0.0 

    print("\n-------------------------------------------------------------------------")
    print(f"  AP  (Average Performance) : {average_performance * 100:.2f}% ")
    print(f"  BWT (Backward Transfer)   : {backward_transfer * 100:+.2f}% ")
    print("-------------------------------------------------------------------------")

if __name__ == '__main__':
    run_continual_experiment()

