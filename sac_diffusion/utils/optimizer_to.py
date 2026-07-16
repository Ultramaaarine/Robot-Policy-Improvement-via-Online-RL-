import torch


def optimizer_to(optimizer:torch.optim.Optimizer,device):
        
        for state in optimizer.state.values():
            for k, v in state.items():
             if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
        return optimizer 