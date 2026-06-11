import torch
import torch.nn as nn

class GradientInterceptor:
    def __init__(self, alpha_base=0.8, alpha_min=0.1, gamma_topo=1.0):
        self.alpha_base = alpha_base
        self.alpha_min = alpha_min
        self.gamma_topo = gamma_topo

    def get_feature_interceptor(self, memory_bank, layer_key, param_type='b'):
        def hook(grad):
            if grad is None: return grad
            U_global = memory_bank.get_feat_basis(layer_key)
            if U_global.size(0) == 0 or U_global.size(1) == 0: return grad
            
            device = grad.device
            U_global = U_global.to(device)

            r_dim = memory_bank.r
            saturation = U_global.size(1) / float(r_dim)
            alpha = max(self.alpha_base * (1.0 - saturation), self.alpha_min)
            
            if param_type == 'b':
                historical_projection = torch.matmul(U_global, torch.matmul(U_global.t(), grad))
                return grad - alpha * historical_projection
            elif param_type == 'a':
                historical_projection = torch.matmul(torch.matmul(grad, U_global), U_global.t())
                return grad - alpha * historical_projection
            return grad
        return hook

    def get_topology_interceptor(self, memory_bank, layer_key):
        def hook(grad):
            if grad is None: return grad
            
            Sigma_topo = memory_bank.get_topo_matrix(layer_key)
            if torch.all(Sigma_topo == 0): return grad
            
            device = grad.device
            Sigma_topo = Sigma_topo.to(device)
            
            max_energy = torch.max(torch.abs(Sigma_topo)) + 1e-8
            safe_Sigma = Sigma_topo / max_energy
            
            M_topo = torch.linalg.matrix_exp(-self.gamma_topo * safe_Sigma)
            
            return torch.matmul(M_topo, grad)
        return hook

def hook_system_components(gnn2, memory_bank, interceptor):
    handles = []
    
    for name, param in gnn2.named_parameters():
        if param.requires_grad:
            layer_key = ".".join(name.split('.')[:-2]) if '.' in name else name
            
            if 'lin_feat_b' in name:
                feat_hook_fn = interceptor.get_feature_interceptor(memory_bank, layer_key, param_type='b')
                handles.append(param.register_hook(feat_hook_fn))
                
            elif 'lin_feat_a' in name:
                feat_hook_fn = interceptor.get_feature_interceptor(memory_bank, layer_key, param_type='a')
                handles.append(param.register_hook(feat_hook_fn))

            elif 'lin_struct_b' in name:
                topo_hook_fn = interceptor.get_topology_interceptor(memory_bank, layer_key)
                handles.append(param.register_hook(topo_hook_fn))
                
    return handles
