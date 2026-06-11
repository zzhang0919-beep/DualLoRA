import torch
import torch.nn as nn

class MemoryBank:
    def __init__(self, r=4):
        self.r = r

        self.U_feat_dict = {}
        
        self.Sigma_topo_dict = {}

    def to(self, device):
        for k, v in self.U_feat_dict.items():
            if isinstance(v, torch.Tensor):
                self.U_feat_dict[k] = v.to(device)
                
        for k, v in self.Sigma_topo_dict.items():
            if isinstance(v, torch.Tensor):
                self.Sigma_topo_dict[k] = v.to(device)
                
        return self

    @torch.no_grad()
    def update_feature_bank(self, layer_key, z_feat, svd_threshold=1e-4):
        if z_feat is None:
            return
            
        device = z_feat.device

        Sigma_f = torch.matmul(z_feat.t(), z_feat) 
        
        U_new, S_new, _ = torch.linalg.svd(Sigma_f, full_matrices=False)
        valid_mask_new = S_new > svd_threshold
        if valid_mask_new.sum() == 0:
            return
        U_new_valid = U_new[:, valid_mask_new] 

        if layer_key not in self.U_feat_dict or self.U_feat_dict[layer_key].size(1) == 0:
            self.U_feat_dict[layer_key] = U_new_valid.to(device)
            return

        U_old = self.U_feat_dict[layer_key].to(device) 
        
        projection = U_old @ (U_old.t() @ U_new_valid)
        delta_U = U_new_valid - projection
        
        if torch.norm(delta_U) < 1e-7:
            return
            
        U_pure, S_pure, _ = torch.linalg.svd(delta_U, full_matrices=False)
        valid_mask_pure = S_pure > svd_threshold
        
        if valid_mask_pure.sum() > 0:
            U_pure_valid = U_pure[:, valid_mask_pure]
            combined_U = torch.cat([U_old, U_pure_valid], dim=1)
            self.U_feat_dict[layer_key] = combined_U[:, :self.r]

    @torch.no_grad()
    def update_topology_bank(self, layer_key, z_struct, target_adj, train_mask):
        if z_struct is None or target_adj is None:
            return
            
        device = z_struct.device

        train_idx = torch.where(train_mask)[0]
        N_train = len(train_idx)
        if N_train <= 1:
            return
            
        Z_sub = z_struct[train_idx]
        
        A_sub = target_adj[train_idx][:, train_idx].float().to(device)
        D_sub_vec = torch.sum(A_sub, dim=1)

        D_sub_vec = torch.where(D_sub_vec == 0, torch.ones_like(D_sub_vec), D_sub_vec)
        D_inv_sqrt = torch.diag(torch.pow(D_sub_vec, -0.5))
        L_sub = torch.eye(N_train, device=device) - (D_inv_sqrt @ A_sub @ D_inv_sqrt)
        
        Sigma_new = Z_sub.t() @ L_sub @ Z_sub
        
        s_vals = torch.linalg.svdvals(Sigma_new)
        if len(s_vals) > 0 and s_vals[0] > 1e-6:
            Sigma_new = Sigma_new / s_vals[0]
            
        if layer_key not in self.Sigma_topo_dict:
            self.Sigma_topo_dict[layer_key] = torch.zeros((self.r, self.r), device=device)
            
        self.Sigma_topo_dict[layer_key] += Sigma_new

    def get_feat_basis(self, layer_key):
        if layer_key in self.U_feat_dict:
            return self.U_feat_dict[layer_key]
        return torch.zeros((self.r, 0))

    def get_topo_matrix(self, layer_key):
        if layer_key in self.Sigma_topo_dict:
            return self.Sigma_topo_dict[layer_key]
        return torch.zeros((self.r, self.r))
