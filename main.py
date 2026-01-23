import torch
from torch.distributions import Categorical
import gymnasium as gym
import wandb

from models import Actor, Critic


test_name = "test10"
episodes = 1000
gamma = 0.99
actor_learning_rate = 0.0005
critic_learning_rate = 0.001
n_steps = 10

env = gym.make("LunarLander-v3")

actor = Actor(env.observation_space.shape[0], env.action_space.n).to("cuda")
critic = Critic(env.observation_space.shape[0]).to("cuda")
# actor.load_state_dict(torch.load(f"checkpoints/test1_actor.pth"))
# critic.load_state_dict(torch.load(f"checkpoints/test1_critic.pth"))

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_learning_rate)

wandb.init(project="LunarLander-v3", name=test_name)


def bp(V, target, log_prob, ):
    value_loss = torch.nn.functional.mse_loss(V, target.detach())
    value_loss.backward()

    actor_loss = - (target - V).detach() * log_prob
    actor_loss.backward()


def step():
    critic_optimizer.step()
    critic_optimizer.zero_grad()
    actor_optimizer.step()
    actor_optimizer.zero_grad()


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

        observations.insert(0, obs)
        rewards.insert(0, reward)
        log_probs.insert(0, dist.log_prob(action))

        if terminated:
            sum_reward = 0
            for reward, obs, log_prob in zip(rewards, observations[1:], log_probs):
                sum_reward *= gamma
                sum_reward += reward
                sum_reward_tensor = torch.tensor(sum_reward, dtype=torch.float32).to("cuda")
                V = critic(torch.tensor(obs).to("cuda"))
                bp(V, sum_reward_tensor, log_prob)
            step()
            break
        else:
            if truncated or steps % n_steps == 0:
                V_last = critic(torch.tensor(observations[0]).to("cuda"))
                sum_reward = 0
                for reward, obs, log_prob in zip(rewards, observations[1:], log_probs):
                    sum_reward *= gamma
                    sum_reward += reward
                    V_last *= gamma
                    sum_reward_tensor = torch.tensor(sum_reward, dtype=torch.float32).to("cuda")
                    V = critic(torch.tensor(obs).to("cuda"))
                    bp(V, sum_reward_tensor + V_last, log_prob)
                step()
                if truncated:
                    break
                observations = [observations[0]]
                rewards = []
                log_probs = []

    wandb.log({
        "reward": total_reward,
        "steps": steps,
    })
    print(f"Episode: {episode}, Steps: {steps}, Total reward: {total_reward}")

torch.save(actor.state_dict(), f"checkpoints/{test_name}_actor.pth")
torch.save(critic.state_dict(), f"checkpoints/{test_name}_critic.pth")
env.close()
wandb.finish()
