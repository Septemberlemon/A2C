import torch
from torch.distributions import Categorical
import gymnasium as gym
import wandb

from models import Actor, Critic


test_name = "all_0.1_entropy"
episodes = 1000
gamma = 0.99
actor_learning_rate = 0.0005
critic_learning_rate = 0.001

env = gym.make("LunarLander-v3")

actor = Actor(env.observation_space.shape[0], env.action_space.n).to("cuda")
critic = Critic(env.observation_space.shape[0]).to("cuda")
# actor.load_state_dict(torch.load(f"checkpoints/test1_actor.pth"))
# critic.load_state_dict(torch.load(f"checkpoints/test1_critic.pth"))

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project="LunarLander-v3", name=test_name)

for episode in range(episodes):
    obs, info = env.reset()
    steps = 0
    total_reward = 0
    while True:
        obs_tensor = torch.tensor(obs).to("cuda")
        logits = actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        new_obs, reward, terminated, truncated, info = env.step(action.item())

        new_obs_tensor = torch.tensor(new_obs).to("cuda")
        V = critic(obs_tensor)
        next_V = critic(new_obs_tensor)
        Q = torch.tensor(reward / 100, dtype=torch.float32).to("cuda")
        if not terminated:
            Q += gamma * next_V.detach()
        A = Q - V.detach()

        value_loss = torch.nn.functional.mse_loss(V, Q)
        critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
        critic_optimizer.step()

        entropy = dist.entropy().mean()
        actor_loss = - A * dist.log_prob(action) - 0.1 * entropy
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
        actor_optimizer.step()

        obs = new_obs

        steps += 1
        total_reward += reward

        if terminated or truncated:
            break

    wandb.log({
        "reward": total_reward,
        "steps": steps,
    })
    print(f"Episode: {episode}, Steps: {steps}, Total reward: {total_reward}")

torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
env.close()
wandb.finish()
