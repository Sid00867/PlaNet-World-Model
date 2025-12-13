import numpy as mp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment_variables import *

class rssm(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # For 28x28, this will be 128 * 3 * 3 = 1152
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            self.embed_dim = self.obs_encoder(dummy).shape[1]

        # Posterior heads
        concat_dim = self.embed_dim + deterministic_dim
        self.mu = nn.Linear(concat_dim, latent_dim)
        self.log_sigma = nn.Linear(concat_dim, latent_dim)

        # gru
        self.gru = nn.GRUCell(latent_dim + action_dim, deterministic_dim)

        #decoder
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 128 * 3 * 3),
            nn.Dropout(0.3)
        )

        self.decoder = nn.Sequential(
            # Input: 128 x 3 x 3
            nn.ConvTranspose2d(128,128,3,1,1), nn.ReLU(),       
            nn.ConvTranspose2d(128,64,4,2,1, output_padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(64,32,4,2,1),   nn.ReLU(),       
            nn.ConvTranspose2d(32,3,4,2,1),                     
            nn.Sigmoid()
        )

        # reward model 
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + deterministic_dim, 256),
            nn.ELU(), 
            nn.Linear(256, 1)
        )

        # prior model
        self.prior_fc = nn.Sequential(
            nn.Linear(deterministic_dim, 256),
            nn.ELU(), 
            nn.Linear(256, 128),
        )

        self.prior_mu = nn.Linear(128, latent_dim)
        self.prior_log_sigma = nn.Linear(128, latent_dim)


    def forward_train(self, h_prev, s_prev, a_prev, o_embed):
        # gru
        gru_input = torch.cat([s_prev, a_prev], dim=-1)   
        h_t = self.gru(gru_input, h_prev)                    

        # posterior
        concat_post = torch.cat([o_embed, h_t], dim=-1)  
        mu_post = self.mu(concat_post)                      
        log_sigma_post = self.log_sigma(concat_post)        
        sigma_post = torch.exp(log_sigma_post)
        s_t_post = mu_post + sigma_post * torch.randn_like(sigma_post)  

        # prior
        prior_hidden = self.prior_fc(h_t)                    
        mu_prior = self.prior_mu(prior_hidden)               
        log_sigma_prior = self.prior_log_sigma(prior_hidden) 

        # decoder
        dec_input = torch.cat([s_t_post, h_t], dim=-1)       
        x = self.fc(dec_input)                               
        x = x.view(-1, 128, 3, 3)                         
        o_recon = self.decoder(x)                          

        # reward
        reward_input = torch.cat([s_t_post, h_t], dim=-1)   
        reward_pred = self.reward_model(reward_input).squeeze(-1)

        return (mu_post, log_sigma_post), (mu_prior, log_sigma_prior), o_recon, reward_pred, h_t, s_t_post

    #use rssm, actor model to imagine from input state to a set horzion, return state and action sequences
    def imagine_rollout(self, actor, start_h, start_s, horizon):
        h = start_h
        s = start_s
        
        # store the dream
        h_seq = []
        s_seq = []
        action_seq = []
        
        for t in range(horizon):
            state_features = torch.cat([s, h], dim=-1)
            action_logits = actor(state_features.detach()) #  stopping gradients calculation here coz
            #we do not want to accidentally train the world model to make the world easier for the actor (while training actor) (actor model is indpenedent of world model)

            if self.training:
                action_prob = F.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.OneHotCategorical(probs=action_prob)
                action = action_dist.sample() # convert action logits to probs, sample one action based on thos probs, make one hot vec

                action = action + (action_prob - action_prob.detach()) # during forward pass, both action_probs would cancel each other out 
                #leaving only action for in the simulation. but while optimizing (backpropagating), pytorch doesnt see action cause it sampled and 
                #one cannot find a gradient of a sample operation, action_prob.detach() would withdraw itself from gradient caluculation bcoz of .detach(),
                #leaving the gradients of only action prob. The Actor learns as if it had passed the soft probabilities to the world model.
                # It learns: "If I had slightly increased the probability of [0], the loss would have gone down."
            else:
                action = F.one_hot(action_logits.argmax(-1), action_dim).float() 

            gru_input = torch.cat([s, action], dim=-1)
            h = self.gru(gru_input, h)
            
            prior_hidden = self.prior_fc(h)
            mu = self.prior_mu(prior_hidden)
            s = mu 
            
            h_seq.append(h)
            s_seq.append(s)
            action_seq.append(action)
            
        return torch.stack(h_seq), torch.stack(s_seq), torch.stack(action_seq)
    