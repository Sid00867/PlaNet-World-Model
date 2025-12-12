import torch
import torch.nn.functional as F
from environment_variables import *
from metrics_hooks import log_training_step
from rssm import rssm

rssmmodel = rssm().to(DEVICE)
optimizer = torch.optim.Adam(rssmmodel.parameters(), lr=learnrate)

def compute_psnr(x, x_hat):
    mse = F.mse_loss(x, x_hat, reduction="mean")
    return 10.0 * torch.log10(1.0 / mse)

def compute_loss(o_t, o_embed, a_t, r_t, h_prev, s_prev):
    (mu_post, log_sigma_post), (mu_prior, log_sigma_prior), o_recon, reward_pred, h_t, s_t = rssmmodel.forward_train(
        h_prev, s_prev, a_t, o_embed
    )

    recon_loss  = F.mse_loss(o_recon, o_t, reduction='mean')
    reward_loss = F.mse_loss(reward_pred, r_t.squeeze(-1), reduction='mean')

    log_sigma_post  = torch.clamp(log_sigma_post,  -log_sigma_clamp, log_sigma_clamp)
    log_sigma_prior = torch.clamp(log_sigma_prior, -log_sigma_clamp, log_sigma_clamp)

    kl_loss = torch.mean(
        torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_post,  torch.exp(log_sigma_post)),
            torch.distributions.Normal(mu_prior, torch.exp(log_sigma_prior))
        )
    )

    total_loss = recon_loss + reward_loss + beta * kl_loss
    psnr = compute_psnr(o_t, o_recon)

    return total_loss, recon_loss, kl_loss, reward_loss, psnr, h_t, s_t


def train_sequence(C, dataset, batch_size, seq_len):

    rssmmodel.train()

    for step in range(C):

        o_t, a_t, r_t, _ = dataset.sample(batch_size, seq_len)
        B = o_t.size(0)

        #parallelize encoding
        flat_obs = o_t.view(-1, *obs_shape)
        flat_embed = rssmmodel.obs_encoder(flat_obs)
        embed_t = flat_embed.view(B, seq_len, -1)
        
        # The RSSM needs "Prev Action" to predict "Current State".
        # We shift actions right by 1, padding the start with zeros.
        shifted_actions = torch.cat([
            torch.zeros(B, 1, action_dim, device=DEVICE), # Dummy action for first step
            a_t[:, :-1, :]                                # Shifted actions
        ], dim=1)

        h_t = torch.zeros(B, deterministic_dim, device=DEVICE)
        s_t = torch.zeros(B, latent_dim, device=DEVICE) 

        total_loss_accum = 0
        total_recon = 0
        total_kl = 0
        total_reward = 0
        total_psnr = 0

        optimizer.zero_grad()

        for L in range(seq_len):
            
            (
                steploss,
                recon_loss,
                kl_loss,
                reward_loss,
                psnr,
                h_t,
                s_t  
            ) = compute_loss(
                o_t=o_t[:, L],          
                o_embed=embed_t[:, L], 
                a_t=shifted_actions[:, L], 
                r_t=r_t[:, L],
                h_prev=h_t,
                s_prev=s_t
            )

            total_loss_accum += steploss
            total_recon      += recon_loss
            total_kl         += kl_loss
            total_reward     += reward_loss
            total_psnr       += psnr

        total_loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(rssmmodel.parameters(), grad_clipping_value)
        optimizer.step()

        log_training_step(
            total_loss    = (total_loss_accum / seq_len).item(),
            recon_loss    = (total_recon   / seq_len).item(),
            kl_loss       = (total_kl      / seq_len).item(),
            reward_loss   = (total_reward / seq_len).item(),
            psnr          = (total_psnr   / seq_len).item(),
        )