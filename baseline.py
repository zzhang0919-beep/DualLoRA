import torch
import torch.nn.functional as F
from util import batched_smmd_loss, batched_gct_loss  

def train_single_task_baseline(
    model, projector, logreg, 
    global_dataset, train_mask,
    pretrain_graph_loader, SMMD, ppr_weight, mask,
    target_adj,
    device, args
):
    model.train()
    projector.train()
    logreg.train()

    target_adj_safe = torch.clamp(target_adj, 0.0, 1.0)

    trainable_params = [
        {"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1}, 
        {"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2}, 
        {"params": filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr3, 'weight_decay': args.wd3}
    ]
    optimizer = torch.optim.Adam(trainable_params)

    pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device) 
    weight_tensor[weight_mask] = pos_weight

    for epoch in range(args.num_epochs):
        optimizer.zero_grad()

        feature_map = projector(global_dataset.x)
        emb, emb1, emb2 = model(feature_map, global_dataset.edge_index)
        
        logits = logreg(emb)
        train_logits = logits[train_mask]
        train_labels = global_dataset.y[train_mask]

        cls_loss = F.cross_entropy(train_logits, train_labels, ignore_index=-1)

        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)

        emb1_norm = F.normalize(emb1, p=2, dim=1, eps=1e-8)
        emb2_norm = F.normalize(emb2, p=2, dim=1, eps=1e-8)
        emb1_norm = torch.nan_to_num(emb1_norm, nan=0.0)
        emb2_norm = torch.nan_to_num(emb2_norm, nan=0.0)
        ct_loss = 0.5 * (batched_gct_loss(emb1_norm, emb2_norm, 1000, mask, args.tau) + 
                         batched_gct_loss(emb2_norm, emb1_norm, 1000, mask, args.tau)).mean()

        rec_adj_raw = torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T)
        rec_adj = torch.sigmoid(rec_adj_raw)
        if torch.isnan(rec_adj).any() or torch.isinf(rec_adj).any():
            rec_adj = torch.nan_to_num(rec_adj, nan=0.5, posinf=0.99, neginf=0.01)
        rec_adj = torch.clamp(rec_adj, min=1e-6, max=1.0 - 1e-6)
        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj_safe.view(-1), weight=weight_tensor)

        if torch.isnan(cls_loss) or torch.isinf(cls_loss): cls_loss = torch.tensor(0.0, device=device)
        if torch.isnan(smmd_loss_f) or torch.isinf(smmd_loss_f): smmd_loss_f = torch.tensor(0.0, device=device)
        if torch.isnan(ct_loss) or torch.isinf(ct_loss): ct_loss = torch.tensor(0.0, device=device)
        if torch.isnan(loss_rec) or torch.isinf(loss_rec): loss_rec = torch.tensor(0.0, device=device)

        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f + args.l3 * ct_loss + args.l4 * loss_rec
        
        loss.backward()

        all_trainable_tensors = [p for p in model.parameters() if p.requires_grad] + \
                                list(projector.parameters()) + \
                                list(logreg.parameters())
        torch.nn.utils.clip_grad_norm_(all_trainable_tensors, max_norm=2.0)

        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            acc = (train_logits.argmax(dim=1) == train_labels).float().mean()
            print(f"Baseline | Epoch [{epoch+1}/{args.num_epochs}] -> Train Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%")