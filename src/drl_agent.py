from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        memory_size: int = 12000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.20,
        epsilon_decay: float = 0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=memory_size)

        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: Optional[np.ndarray], done: bool) -> None:
        if next_state is None:
            next_state = np.zeros(self.state_size, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * (1 - dones_t) * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


def train_dqn_agent(
    env,
    episodes: int = 200,
    device: Optional[torch.device] = None,
    save_path: str | Path = "outputs/dqn_policy.pth",
    verbose_every: int = 10,
):
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, device=device)
    reward_history: List[float] = []
    best_reward = -float("inf")

    print(f"Starting DQN training for {episodes} episodes on {env.n_steps} validation steps...")

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        action_counts = np.zeros(env.action_size, dtype=np.int64)

        while not done:
            action = agent.select_action(state)
            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = np.zeros_like(state) if next_state is None else next_state
            total_reward += reward

        agent.decay_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()

        reward_history.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path)

        if episode == 1 or episode % verbose_every == 0 or episode == episodes:
            recent_window = reward_history[-min(len(reward_history), verbose_every) :]
            recent_avg = float(np.mean(recent_window)) if recent_window else total_reward
            action_ratio = action_counts / max(1, action_counts.sum())
            action_ratio_text = ", ".join(
                f"a{idx}:{ratio:.2f}" for idx, ratio in enumerate(action_ratio.tolist())
            )
            print(
                f"[DQN] Episode {episode:03d}/{episodes} | "
                f"reward={total_reward:.3f} | avg_recent={recent_avg:.3f} | epsilon={agent.epsilon:.3f} | "
                f"action_mix=[{action_ratio_text}]"
            )

    agent.load(save_path)
    print("DQN training complete.")
    return agent, reward_history


def evaluate_policy(agent: DQNAgent, env) -> Dict[str, np.ndarray | list]:
    state = env.reset()
    done = False
    actions, predictions, rewards, under_flags = [], [], [], []

    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        actions.append(action)
        predictions.append(info["selected_prediction"])
        rewards.append(reward)
        under_flags.append(info["under_prediction"])
        state = np.zeros_like(state) if next_state is None else next_state

    return {
        "actions": np.asarray(actions, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "under_prediction_flags": np.asarray(under_flags, dtype=bool),
    }
