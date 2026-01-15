from collections.abc import Callable, Iterable
from typing import Optional, List
import torch
import math
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    """
    Two dictionary object:
        - defaults: 
            Stores the default hyperparamters value for parameters groups if 
            they did not explicitly define them.
        
        - self.state[p]: 
            Store the value that will be used by the optimizer to updates the 
            parameters group p, constantly changing as we updates the value of p.
    """
    def __init__(
            self, params,
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # Create the default hyperparamters value for different para_group
        defaults = {
            "lr": lr,
            "lambda": weight_decay,
            "eps": eps,
            "b1": betas[0],
            "b2": betas[1]
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # Get all hyperparas
            lr = group["lr"] 
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]
            lbd = group["lambda"]

            for p in group["params"]:
                """Iterate over each para in the group"""
                if p.grad is None:
                    continue

                # Get state associated with the curr parameter group p
                state = self.state[p] 
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                t = state.get("t", 1) 
                m = state["m"]
                v = state["v"]
                
                # Get the gradient of loss with respect to p.
                grad = p.grad.data 

                # Update 1st & 2nd Moments Est.
                new_m = m*b1 + (1-b1)*grad
                new_v= v*b2 + (1-b2)*grad**2
                state["m"] = new_m
                state["v"] = new_v

                # Update weights and weights decay
                lr_t = lr*(math.sqrt(1-(b2)**t)/(1-(b1)**t))
                p.data -= lr_t * new_m / (torch.sqrt(new_v)+eps)
                p.data -= lr * lbd * p.data

                # Increment iteration number.
                state["t"] = t + 1 
        return loss
    


def lr_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_aiters: int,
):
    """
    Return the learning alpha_t at time t.
    """
    if it < warmup_iters:
        return it/warmup_iters*max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_aiters:
        return min_learning_rate+0.5*(
            1+math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_aiters-warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
    

def grad_clip(paras:List[nn.Parameter], max_norm: float, eps=1e-6):
    """
    Apply Gradient clipping on all parameters, clip to max_norm.
    """
    grad_l2_squared = 0.0
    for p in paras:
        if p.grad is None:
            continue
        grad_l2_squared += p.grad.detach().pow(2).sum().item()

    grad_l2 = math.sqrt(grad_l2_squared)
    if grad_l2 > max_norm:
        for p in paras:
            if p.grad is None:
                continue
            p.grad *= (max_norm/(grad_l2+eps))
    
    
    return max(grad_l2, max_norm)
            
