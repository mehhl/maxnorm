# A wrapper to add max_norm weight correction capability to PyTorch SGD optimizer.

import torch

class MNSGD(torch.optim.SGD):    
    def step(self):
        super().step()
        with torch.no_grad():
            # apply the max_norm weight correction per group of parameters
            for group in self.param_groups:
                # rescale iff group has specified max_norm
                if group.get('max_norm'):
                    for tensor in group['params']:
                        if tensor.dim() > 1:
                            torch.renorm(tensor, p=2, dim=0, maxnorm=group['max_norm'], out=tensor)
