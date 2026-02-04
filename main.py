import torch
from torch.distributions import Categorical
import gymnasium as gym
import wandb
import numpy as np

from models import Actor, Critic


test_name = "test16"
episodes = 1000
gamma = 0.99
actor_learning_rate = 0.0005
critic_learning_rate = 0.001
n_steps = 100

env = gym.make("LunarLander-v3")

actor = Actor(env.observation_space.shape[0], env.action_space.n).to("cuda")
critic = Critic(env.observation_space.shape[0]).to("cuda")
# actor.load_state_dict(torch.load(f"checkpoints/test15_actor.pth"))
# critic.load_state_dict(torch.load(f"checkpoints/test15_critic.pth"))

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project="LunarLander-v3", name=test_name)


def update(V_s, targets, log_probs, entropy, entropy_coef):
    value_loss = torch.nn.functional.huber_loss(V_s, targets.detach())

    advantage = (targets - V_s).detach()
    if advantage.size(0) != 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
    actor_loss = - (advantage * log_probs).mean() - entropy_coef * entropy.mean()

    critic_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()


for episode in range(episodes):
    obs, info = env.reset()
    steps = 0
    total_reward = 0
    observations = [obs]
    rewards = []
    log_probs = []
    entropy_coef = 0.01 * (1 - episode / episodes)
    entropies = []
    while True:
        obs_tensor = torch.tensor(obs).to("cuda")
        logits = actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        obs, reward, terminated, truncated, info = env.step(action.item())
        steps += 1
        total_reward += reward

        observations.append(obs)
        rewards.append(reward / 100)
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        if terminated:
            sum_reward = 0
            sum_rewards = []
            for reward, log_prob in zip(rewards[::-1], log_probs[::-1]):
                sum_reward *= gamma
                sum_reward += reward
                sum_rewards.append(sum_reward)
            sum_rewards_tensor = torch.tensor(sum_rewards[::-1], dtype=torch.float32).to("cuda")
            V_s = critic(torch.from_numpy(np.stack(observations[:-1])).to("cuda"))
            update(V_s, sum_rewards_tensor, torch.stack(log_probs), torch.stack(entropies), entropy_coef)
            break
        else:
            if truncated or steps % n_steps == 0:
                V_last = critic(torch.tensor(observations[-1]).to("cuda")).item()
                sum_reward = 0
                targets = []
                for reward, log_prob in zip(rewards[::-1], log_probs[::-1]):
                    sum_reward *= gamma
                    sum_reward += reward
                    V_last *= gamma
                    targets.append(sum_reward + V_last)
                targets = torch.tensor(targets[::-1], dtype=torch.float32).to("cuda")
                V_s = critic(torch.from_numpy(np.stack(observations[:-1])).to("cuda"))
                update(V_s, targets, torch.stack(log_probs), torch.stack(entropies), entropy_coef)
                if truncated:
                    break
                observations = [observations[-1]]
                rewards = []
                log_probs = []
                entropies = []

    wandb.log({
        "reward": total_reward,
        "steps": steps,
    })
    print(f"Episode: {episode}, Steps: {steps}, Total reward: {total_reward}")

torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
env.close()
wandb.finish()
