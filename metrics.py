# Central storage + access API for training / planning convergence metrics

import numpy as np
from collections import deque
from environment_variables import *
import matplotlib.pyplot as plt

class RollingStats:
    def __init__(self, window=metrics_storage_window):
        self.window = window
        self.buffer = deque(maxlen=window)

    def add(self, value):
        self.buffer.append(float(value))

    def mean(self):
        if not self.buffer:
            return 0.0
        return float(np.mean(self.buffer))

    def std(self):
        if not self.buffer:
            return 0.0
        return float(np.std(self.buffer))

    def delta(self):
        """
        Measures change between first and second half of window.
        Useful for detecting plateaus.
        """
        n = len(self.buffer)
        if n < 4:
            return np.inf

        half = n // 2
        old = np.mean(list(self.buffer)[:half])
        new = np.mean(list(self.buffer)[half:])
        return abs(new - old)


class ExperimentMetrics:
    """
    One instance of this object stores everything
    we need to monitor convergence of the outer loop.
    """

    def __init__(self):

        # --- world model metrics ---
        self.world_loss   = RollingStats(window=metrics_storage_window)
        self.recon_mse    = RollingStats(window=metrics_storage_window)
        self.kl_loss      = RollingStats(window=metrics_storage_window)
        self.reward_loss = RollingStats(window=metrics_storage_window)

        # --- dreamer metrics (NEW) ---
        self.actor_loss   = RollingStats(window=metrics_storage_window)
        self.critic_loss  = RollingStats(window=metrics_storage_window)

        # --- visual quality ---
        self.psnr = RollingStats(window=small_metric_window)

        # --- control performance ---
        self.episode_returns = RollingStats(window=small_metric_window)
        self.success_rate    = RollingStats(window=small_metric_window)
        self.episode_lengths = RollingStats(window=small_metric_window)

        # --- general counters ---
        self.env_steps    = 0
        self.train_steps = 0
        self.episodes    = 0


    def add_world_loss(self, total, recon=None, kl=None, reward=None):
        self.world_loss.add(total)

        if recon is not None:
            self.recon_mse.add(recon)

        if kl is not None:
            self.kl_loss.add(kl)

        if reward is not None:
            self.reward_loss.add(reward)

    def add_actor_critic_loss(self, actor, critic):
        self.actor_loss.add(actor)
        self.critic_loss.add(critic)

    def add_psnr(self, value):
        self.psnr.add(value)

    def add_episode_result(self, episode_return, success, length):
        self.episodes += 1
        self.episode_returns.add(episode_return)
        self.success_rate.add(float(success))
        self.episode_lengths.add(length)

    def step_env(self, num_steps=1):
        self.env_steps += int(num_steps)

    def step_train(self):
        self.train_steps += 1


    def get_means(self):
        """
        Returns a dictionary of commonly logged aggregates.
        """
        return {
            "loss_total": self.world_loss.mean(),
            "loss_recon": self.recon_mse.mean(),
            "loss_kl": self.kl_loss.mean(),
            "loss_reward": self.reward_loss.mean(),
            "loss_actor": self.actor_loss.mean(),   # NEW
            "loss_critic": self.critic_loss.mean(), # NEW
            "psnr": self.psnr.mean(),
            "return": self.episode_returns.mean(),
            "success_rate": self.success_rate.mean(),
            "episode_length": self.episode_lengths.mean(),
            "env_steps": self.env_steps,
            "train_steps": self.train_steps,
            "episodes": self.episodes,
        }


    def get_deltas(self):
        """
        Change metrics useful for convergence detection.
        """
        return {
            "loss_delta": self.world_loss.delta(),
            "recon_delta": self.recon_mse.delta(),
            "psnr_delta": self.psnr.delta(),
            "return_delta": self.episode_returns.delta(),
            "success_delta": self.success_rate.delta(),
        }


    # -------------------------------------------------
    # Convergence check
    # -------------------------------------------------

    def has_converged(
        self,
        loss_eps=loss_eps,
        recon_eps=recon_eps,
        psnr_eps=psnr_eps,
        min_success=min_success,
        min_steps=min_steps,
        max_steps=max_steps
    ):
        if self.env_steps > max_steps:
            return True
        if self.env_steps < min_steps:
            return False

        deltas = self.get_deltas()

        loss_flat  = deltas["loss_delta"]  < loss_eps
        recon_flat = deltas["recon_delta"] < recon_eps
        psnr_flat  = deltas["psnr_delta"]  < psnr_eps

        success_good = self.success_rate.mean() >= min_success

        return loss_flat and recon_flat and psnr_flat and success_good
    
    #plotting

    def _plot_series(self, ax, data, title, ylabel, smooth=1, color=None):
        if len(data) == 0:
            ax.set_title(title + " (no data)")
            return

        series = np.array(data)
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            series = np.convolve(series, kernel, mode="valid")

        ax.plot(series, color=color)
        ax.set_title(title)
        ax.set_xlabel("updates")
        ax.set_ylabel(ylabel)
        ax.grid(True)


    def plot_metrics(self): # Changed to method for easier access if needed, or keep separate function
        pass

def plot_metrics(metrics):
    """
    Displays dashboard of major training curves.
    Expanded to 4x2 grid to include Actor/Critic.
    """

    fig, axes = plt.subplots(4, 2, figsize=(12, 16)) # Taller figure
    fig.suptitle("Dreamer Training Metrics", fontsize=14)

    # Row 1: World Model
    metrics._plot_series(axes[0,0], metrics.world_loss.buffer,
                "World Model Total Loss", "loss")
    metrics._plot_series(axes[0,1], metrics.recon_mse.buffer,
                "Reconstruction MSE", "mse")

    # Row 2: Actor / Critic (NEW)
    metrics._plot_series(axes[1,0], metrics.actor_loss.buffer,
                "Actor Loss (Policy)", "loss", color="green")
    metrics._plot_series(axes[1,1], metrics.critic_loss.buffer,
                "Critic Loss (Value)", "mse", color="red")

    # Row 3: Quality & Return
    metrics._plot_series(axes[2,0], metrics.psnr.buffer,
                "PSNR", "dB")
    metrics._plot_series(axes[2,1], metrics.episode_returns.buffer,
                "Episode Return", "return")

    # Row 4: Task Performance
    metrics._plot_series(axes[3,0], metrics.success_rate.buffer,
                "Success Rate", "fraction")
    metrics._plot_series(axes[3,1], metrics.episode_lengths.buffer,
                "Episode Length", "steps")

    plt.tight_layout()
    plt.show()


METRICS = ExperimentMetrics()