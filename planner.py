from environment_variables import *
import torch
from fitter import rssmmodel 

previous_mean = None

def plan(h_t, s_t):
    global previous_mean
    rssmmodel.eval()
    
    with torch.no_grad():
        
        if previous_mean is not None:
            # Shift mean left by 1: [t+1, t+2, ...]
            new_mean = torch.cat([
                previous_mean[1:], 
                torch.zeros((1, action_dim), device=DEVICE)
            ], dim=0)
            mean = new_mean
            std = torch.ones((planning_horizon, action_dim), device=DEVICE) * 0.5 
        else:
            mean = torch.zeros((planning_horizon, action_dim), device=DEVICE)
            std  = torch.ones((planning_horizon, action_dim), device=DEVICE)

        h_init = h_t.expand(candidates, -1).contiguous()   
        s_init = s_t.expand(candidates, -1).contiguous()  

        for _ in range(optimization_iters):
            candidates_actions = torch.normal(
                mean.expand(candidates, -1, -1), 
                std.expand(candidates, -1, -1)
            ) 

            h = h_init
            s = s_init
            rewards = torch.zeros(candidates, device=DEVICE)

            for t in range(planning_horizon):
                softmax_actions = torch.softmax(candidates_actions[:, t, :], dim=1) 
                
                h, s = rssmmodel.imagine_step(h, s, softmax_actions) 
                r_t = rssmmodel.reward(s, h)      
                rewards += r_t

            top_values, top_idx = torch.topk(rewards, K, dim=0)
            top_actions = candidates_actions[top_idx]    

            # Refit
            mean = top_actions.mean(dim=0)                
            std  = top_actions.std(dim=0) + 1e-5   
            
        previous_mean = mean          

        softmax_best_action = torch.softmax(mean[0], dim=-1)       
        best_action = torch.nn.functional.one_hot(softmax_best_action.argmax(dim=-1), num_classes=action_dim).float()
        return best_action