from metrics import METRICS
import torch
import torch.nn.functional as F


def log_environment_step(
    reward,
    episode_len,
    done,
    action_repeat,
    success=False,
):

    METRICS.step_env(action_repeat)

    # If episode finished, record episode results
    if done:
        METRICS.add_episode_result(
            episode_return=reward,
            success=int(success),
            length=episode_len,
        )


def log_training_step(
    total_loss,
    recon_loss,
    kl_loss,
    reward_loss,
    actor_loss=None, 
    critic_loss=None, 
    psnr=None,
):
    """
    Hook to record metrics during world-model training.
    """

    METRICS.add_world_loss(
        total=total_loss,
        recon=recon_loss,
        kl=kl_loss,
        reward=reward_loss,
    )
    
    if actor_loss is not None and critic_loss is not None:
        METRICS.add_actor_critic_loss(actor_loss, critic_loss)

    if psnr is not None:
        METRICS.add_psnr(psnr)

    METRICS.step_train()