import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from memory_bank import MemoryBank
from interceptor import GradientInterceptor, hook_system_components

def train_single_task_cme(task_id, gnn2, projector, logreg, test_dataset, train_mask, target_adj, args, device, memory_bank, interceptor, 
                          batched_smmd_loss, batched_gct_loss, pretrain_graph_loader, SMMD, ppr_weight, mask):
    gnn2 = gnn2.to(device)
    projector = projector.to(device)
    logreg = logreg.to(device)
    memory_bank = memory_bank.to(device)
    
    x_input = test_dataset.x.to(device)
    edge_index_input = test_dataset.edge_index.to(device)
    y_input = test_dataset.y.to(device)
    train_mask = train_mask.to(device)
    target_adj = target_adj.to(device)
    
    trainable_params = [
        {"params": filter(lambda p: p.requires_grad, gnn2.parameters()), "lr": args.lr_lora},
        {"params": projector.parameters(), "lr": args.lr_lora},
        {"params": logreg.parameters(), "lr": args.lr_head}
    ]

    optimizer = torch.optim.Adam(trainable_params, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    lora_handles = hook_system_components(gnn2, memory_bank, interceptor)
    
    gnn2.train()
    projector.train()
    logreg.train()

    target_adj = torch.clamp(target_adj, min=0, max=1)  

    pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device) 
    weight_tensor[weight_mask] = pos_weight
    
    train_labels = y_input[train_mask]

    for epoch in range(0, args.num_epochs):
        optimizer.zero_grad()
        
        feature_map = projector(x_input)
        emb, emb1, emb2 = gnn2(feature_map, edge_index_input)

        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        ct_loss = 0.5 * (batched_gct_loss(emb1_norm, emb2_norm, 1000, mask, args.tau) + 
                         batched_gct_loss(emb2_norm, emb1_norm, 1000, mask, args.tau)).mean()
        
        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)      
        
        logits = logreg(emb)
        train_logits = logits[train_mask]
        
        rec_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))

        if torch.isnan(rec_adj).any() or torch.isinf(rec_adj).any():
            rec_adj = torch.nan_to_num(rec_adj, nan=0.5, posinf=0.99, neginf=0.01)
            
        rec_adj = torch.clamp(rec_adj, min=1e-6, max=1.0 - 1e-6)
        target_adj_safe = torch.clamp(target_adj, min=0.0, max=1.0)

        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj_safe.view(-1), weight=weight_tensor)
        
        cls_loss = loss_fn(train_logits, train_labels)

        if torch.isnan(cls_loss) or torch.isinf(cls_loss):
            cls_loss = torch.tensor(0.0, device=device)

        if torch.isnan(smmd_loss_f) or torch.isinf(smmd_loss_f):
            smmd_loss_f = torch.tensor(0.0, device=device)

        if torch.isnan(ct_loss) or torch.isinf(ct_loss):
            ct_loss = torch.tensor(0.0, device=device)

        if torch.isnan(loss_rec) or torch.isinf(loss_rec):
            loss_rec = torch.tensor(0.0, device=device)
        
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f + args.l3 * ct_loss + args.l4 * loss_rec
        
        loss.backward()

        all_trainable_tensors = [p for p in gnn2.parameters() if p.requires_grad] + \
                                list(projector.parameters()) + \
                                list(logreg.parameters())
                                
        torch.nn.utils.clip_grad_norm_(all_trainable_tensors, max_norm=2.0)

        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            preds = torch.argmax(train_logits, dim=1)
            train_acc = (torch.sum(preds == train_labels).float() / train_labels.shape[0]).item()
            print(f"Task {task_id} | Epoch [{epoch+1}/{args.num_epochs}] -> Train Loss: {loss.item():.4f}, Acc: {train_acc*100:.2f}%")

    for h in lora_handles:
        h.remove()
    lora_handles.clear()

    gnn2.eval()
    projector.eval()
    with torch.no_grad():
        clean_feature = projector(x_input)
        _ = gnn2(clean_feature, edge_index_input)
        
        for name, layer in gnn2.named_modules():
            if hasattr(layer, 'current_z_feat') and hasattr(layer, 'current_z_struct'):
                layer_key = name if name != "" else "gnn2_layer"
                
                memory_bank.update_feature_bank(
                    layer_key=layer_key,
                    z_feat=layer.current_z_feat,
                    svd_threshold=args.svd_threshold
                )
                
                memory_bank.update_topology_bank(
                    layer_key=layer_key,
                    z_struct=layer.current_z_struct,
                    target_adj=target_adj,
                    train_mask=train_mask
                )

    del feature_map, emb, emb1, emb2, optimizer, trainable_params
    gc.collect()
    torch.cuda.empty_cache()


def train_single_task_graphcontrol(task_id, gnn2, projector, logreg, test_dataset, train_mask, target_adj, args, device,
                                  P_prime, batched_smmd_loss, batched_gct_loss, pretrain_graph_loader, SMMD, ppr_weight, mask):
    gnn2 = gnn2.to(device)
    projector = projector.to(device)
    logreg = logreg.to(device)
    
    x_input = test_dataset.x.to(device)
    edge_index_input = test_dataset.edge_index.to(device)
    y_input = test_dataset.y.to(device)
    train_mask = train_mask.to(device)
    target_adj = target_adj.to(device)
    if P_prime is not None:
        P_prime = P_prime.to(device)
    
    trainable_params = [
        {"params": filter(lambda p: p.requires_grad, gnn2.parameters()), "lr": 5e-5},
        {"params": projector.parameters(), "lr": 9e-4},
        {"params": logreg.parameters(), "lr": 5e-2}
    ]
    optimizer = torch.optim.Adam(trainable_params, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    gnn2.train()
    projector.train()
    logreg.train()

    target_adj = torch.clamp(target_adj, min=0, max=1)
    
    pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device)
    weight_tensor[weight_mask] = pos_weight
    
    train_labels = y_input[train_mask]
    
    for epoch in range(0, args.num_epochs):
        optimizer.zero_grad()
        
        feature_map = projector(x_input)
        
        emb, emb1, emb2 = gnn2(feature_map, edge_index_input, P_prime=P_prime)
        
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        ct_loss = 0.5 * (batched_gct_loss(emb1_norm, emb2_norm, 1000, mask, args.tau) + 
                         batched_gct_loss(emb2_norm, emb1_norm, 1000, mask, args.tau)).mean()
        
        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)      
        
        logits = logreg(emb)
        train_logits = logits[train_mask]
        
        rec_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))

        if torch.isnan(rec_adj).any() or torch.isinf(rec_adj).any():
            rec_adj = torch.nan_to_num(rec_adj, nan=0.5, posinf=0.99, neginf=0.01)
            
        rec_adj = torch.clamp(rec_adj, min=1e-6, max=1.0 - 1e-6)
        target_adj_safe = torch.clamp(target_adj, min=0.0, max=1.0)

        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj_safe.view(-1), weight=weight_tensor)
        cls_loss = loss_fn(train_logits, train_labels)

        if torch.isnan(cls_loss) or torch.isinf(cls_loss): cls_loss = torch.tensor(0.0, device=device)
        if torch.isnan(smmd_loss_f) or torch.isinf(smmd_loss_f): smmd_loss_f = torch.tensor(0.0, device=device)
        if torch.isnan(ct_loss) or torch.isinf(ct_loss): ct_loss = torch.tensor(0.0, device=device)
        if torch.isnan(loss_rec) or torch.isinf(loss_rec): loss_rec = torch.tensor(0.0, device=device)
        
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f + args.l3 * ct_loss + args.l4 * loss_rec
        
        loss.backward()

        all_trainable_tensors = [p for p in gnn2.parameters() if p.requires_grad] + \
                                list(projector.parameters()) + \
                                list(logreg.parameters())
                                
        torch.nn.utils.clip_grad_norm_(all_trainable_tensors, max_norm=2.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            preds = torch.argmax(train_logits, dim=1)
            train_acc = (torch.sum(preds == train_labels).float() / train_labels.shape[0]).item()
            print(f"GraphControl Task {task_id} | Epoch [{epoch+1}/{args.num_epochs}] -> Train Loss: {loss.item():.4f}, Acc: {train_acc*100:.2f}%")
            