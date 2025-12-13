import torch
import torch.nn.functional as F
from environment_variables import *
from fitter import actor_net 

#uses the model we're training/trained to explore/for inference by returning action for given state
def plan(h_t, s_t):
    actor_net.eval()
    
    with torch.no_grad():
        state_features = torch.cat([s_t, h_t], dim=-1)
        action_logits = actor_net(state_features)
        action_idx = action_logits.argmax(dim=-1)
        action_onehot = F.one_hot(action_idx, num_classes=action_dim).float()
        
        return action_onehot

def reset_planner():
    pass