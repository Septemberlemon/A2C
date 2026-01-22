import torch
from torch.distributions import Categorical
import gymnasium as gym
import wandb

from models import Actor, Critic


test_name = "test1"
episodes = 1000
gamma = 0.99
actor_learning_rate = 0.0001
critic_learning_rate = 0.0005
n_steps = 2

env = gym.make("LunarLander-v3")

actor = Actor(env.observation_space.shape[0], env.action_space.n).to("cuda")
critic = Critic(env.observation_space.shape[0]).to("cuda")
# actor.load_state_dict(torch.load(f"checkpoints/test1_actor.pth"))
# critic.load_state_dict(torch.load(f"checkpoints/test1_critic.pth"))

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project="LunarLander-v3", name=test_name)


def bp(V, target, log_prob):
    value_loss = torch.nn.functional.mse_loss(V, target.detach())
    critic_optimizer.zero_grad()
    value_loss.backward()
    critic_optimizer.step()

    actor_loss = - (target - V).detach() * log_prob
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


for episode in range(episodes):
    obs, info = env.reset()
    steps = 0
    total_reward = 0
    observations = [obs]
    rewards = []
    log_probs = []
    while True:
        obs_tensor = torch.tensor(obs).to("cuda")
        logits = actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        obs, reward, terminated, truncated, info = env.step(action.item())
        steps += 1
        total_reward += reward

        observations.append(obs)
        rewards.append(reward)
        log_probs.append(dist.log_prob(action))

        if steps >= n_steps:
            if terminated:
                sum_reward = 0
                for i in range(n_steps):
                    sum_reward *= gamma
                    sum_reward += rewards[-i - 1]
                    returns = torch.tensor(sum_reward, dtype=torch.float32).to("cuda")
                    V = critic(torch.tensor(observations[-i - 2]).to("cuda"))
                    bp(V, returns, log_probs[-i - 1])
                break
            elif truncated:
                obs_tensor = torch.tensor(observations[-1]).to("cuda")
                V_last = critic(obs_tensor)
                sum_reward = 0
                for i in range(n_steps):
                    sum_reward *= gamma
                    sum_reward += rewards[-i - 1]
                    returns = torch.tensor(sum_reward, dtype=torch.float32).to("cuda")
                    td_target = returns + gamma ** (i + 1) * V_last
                    V = critic(torch.tensor(observations[-i - 2]).to("cuda"))
                    bp(V, td_target, log_probs[-i - 1])
                break
            else:
                sum_reward = 0
                for reward in rewards[-n_steps:]:
                    sum_reward *= gamma
                    sum_reward += reward
                obs_tensor = torch.tensor(observations[-1]).to("cuda")
                sum_reward_tensor = torch.tensor(sum_reward, dtype=torch.float32).to("cuda")
                td_target = sum_reward_tensor + gamma ** n_steps * critic(obs_tensor)
                V = critic(torch.tensor(observations[-n_steps - 1]).to("cuda"))
                bp(V, td_target, log_probs[-n_steps])

    wandb.log({
        "reward": total_reward,
        "steps": steps,
    })
    print(f"Episode: {episode}, Steps: {steps}, Total reward: {total_reward}")

torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
env.close()
wandb.finish()
