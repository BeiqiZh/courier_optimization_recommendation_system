# src/util/Predictors.py
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import pickle


class Model(ABC):
    def __init__(self):
        super().__init__()
        logging.info('Initializing model')

    @abstractmethod
    def train(self):
        logging.info('Training model')

    @abstractmethod
    def predict(self):
        logging.info('Doing predictions')


class DQNCourierPredictor(Model):
    """
    Deep Q-Network predictor for courier assignment.
    Input:
        feat_df_new, feat_df, df_filtered  – output of DataPreprocessing
        run_configuration                 – namedtuple loaded from JSON
    """

    def __init__(self,
                 feat_df_new: pd.DataFrame,
                 feat_df: pd.DataFrame,
                 df_filtered: pd.DataFrame,
                 run_configuration):
        super().__init__()
        self.feat_df_new = feat_df_new.copy()
        self.feat_df = feat_df.copy()
        self.df_filtered = df_filtered.copy()
        self.cfg = run_configuration

        # ------------------- Hyper-parameters -------------------
        self.state_dim = 3
        self.max_actions = 5 # we always output 5 Q-values (pad with dummy)
        self.hidden = 64
        self.replay_capacity = 10_000
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 0.1
        self.target_update = 10
        self.n_epochs = 15
        self.train_split = 0.85

        # Normalisation stats (computed once)
        self.cost_mean, self.cost_std = None, None
        self.open_mean, self.open_std = None, None

        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Containers for training
        self.states = None
        self.actions = None
        self.rewards = None
        self.courier_ids = None
        self.train_episodes = []
        self.test_episodes = []

    # --------------------------------------------------------------------- #
    # 1. Normalisation helper
    # --------------------------------------------------------------------- #
    def _normalize(self, cost, open_orders, hour):
        return [
            (cost - self.cost_mean) / (self.cost_std + 1e-8),
            (open_orders - self.open_mean) / (self.open_std + 1e-8),
            hour / 23.0
        ]

    # --------------------------------------------------------------------- #
    # 2. Build state matrix
    # --------------------------------------------------------------------- #
    def _prepare_states(self):
        self.cost_mean = self.feat_df_new['cost'].mean()
        self.cost_std = self.feat_df_new['cost'].std()
        self.open_mean = self.feat_df_new['open_orders'].mean()
        self.open_std = self.feat_df_new['open_orders'].std()

        logging.info("Building state matrix")
        self.states = np.array([
            self._normalize(row['cost'], row['open_orders'], row['hour'])
            for _, row in self.feat_df_new.iterrows()
        ])

    # --------------------------------------------------------------------- #
    # 3. Prepare actions, rewards, courier ids
    # --------------------------------------------------------------------- #
    def _prepare_actions_rewards(self):
        logging.info("Preparing actions & rewards")
        self.courier_ids = self.feat_df_new['courier_id'].values

        # action index inside each order (0,1,2,...)
        self.actions = self.feat_df_new.groupby('order_number').cumcount().values

        # rewards according to the rule you defined
        rewards = np.zeros(len(self.feat_df_new))
        for i, row in enumerate(self.feat_df_new.itertuples(index=False)):
            if row.order_assignment_unsuccess_binary == 0:# final courier
                rewards[i] = 1.0 - 0.1 * row.cost
            # synthetic candidate
            elif row.order_assignment_unsuccess_binary == -1:
                rewards[i] = 0.0
            # reassigned / failed
            else:
                rewards[i] = -1.0 - 0.1 * row.cost
        self.rewards = rewards

    # --------------------------------------------------------------------- #
    # 4. Build episodes (one per order)
    # --------------------------------------------------------------------- #
    def _build_episodes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        df = df.reset_index(drop=True)
        episodes = []
        for order, group in tqdm(df.groupby('order_number', sort=False),
                                 desc="Building episodes"):
            idxs = group.index.to_numpy()

            episodes.append({
                'order_number': order,
                'states': self.states[idxs],
                'actions': self.actions[idxs],
                'rewards': self.rewards[idxs],
                'courier_ids': self.courier_ids[idxs],
                'cost': group['cost'].values
            })
        return episodes

    # --------------------------------------------------------------------- #
    # 5. DQN Network
    # --------------------------------------------------------------------- #
    class DQN(nn.Module):
        def __init__(self, state_dim: int, n_actions: int, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )

        def forward(self, x):
            return self.net(x)

    # --------------------------------------------------------------------- #
    # 6. Replay Buffer: A memory bank that stores past experiences,
    # learn from mistakes
    # --------------------------------------------------------------------- #
    class ReplayBuffer:
        def __init__(self, capacity: int, device):
            self.buffer = deque(maxlen=capacity)
            self.device = device  # ← store device

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*batch)
            return (
                torch.FloatTensor(np.array(state)).to(self.device),
                torch.LongTensor(action).to(self.device),
                torch.FloatTensor(reward).to(self.device),
                torch.FloatTensor(np.array(next_state)).to(self.device),
                torch.FloatTensor(done).to(self.device)
            )

        def __len__(self):
            return len(self.buffer)

    # --------------------------------------------------------------------- #
    # 7. DQN Agent
    # --------------------------------------------------------------------- #
    class DQNAgent:
        def __init__(self, model: 'DQNCourierPredictor'):
            self.model = model
            # learns (updated every step)
            self.policy_net = model.DQN(model.state_dim, model.max_actions, model.hidden).to(model.device)
            # stable (updated every 10 steps)
            self.target_net = model.DQN(model.state_dim, model.max_actions, model.hidden).to(model.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
            self.buffer = model.ReplayBuffer(model.replay_capacity, model.device)  # ← pass device
            self.step = 0

        def select_action(self, state):
            #Explore: 10% chance → try random
            # courier
            if random.random() < self.model.epsilon:
                return random.randint(0, self.model.max_actions - 1)
            with torch.no_grad():
                # Exploit: pick courier with highest Q-value
                q = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.model.device))
                return q.argmax().item()

        def train_step(self): #Learning from Memory
            if len(self.buffer) < self.model.batch_size:
                return
            state, action, reward, next_state, done = self.buffer.sample(self.model.batch_size)

            q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
            next_q = self.target_net(next_state).max(1)[0]
            target = reward + self.model.gamma * next_q * (1 - done)

            loss = nn.MSELoss()(q_values, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step += 1
            if self.step % self.model.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    # --------------------------------------------------------------------- #
    # 8. train() – public method
    # --------------------------------------------------------------------- #
    def train(self):
        logging.info("=== Starting DQN training ===")

        # 1. Prepare data
        self._prepare_states()
        self._prepare_actions_rewards()

        # 2. Build episodes from the *real* attempts only
        episodes = self._build_episodes(self.feat_df)

        # 3. Train / test split
        random.shuffle(episodes)
        split_idx = int(self.train_split * len(episodes))
        self.train_episodes = episodes[:split_idx]
        self.test_episodes = episodes[split_idx:]
        logging.info(f"Train episodes: {len(self.train_episodes)}, Test: {len(self.test_episodes)}")

        # === SAVE test_episodes ===
        test_episodes_path = self.cfg.TEST_EPISODES_PATH
        with open(test_episodes_path, 'wb') as f:
            pickle.dump(self.test_episodes, f)
        logging.info(f"Test episodes saved to {test_episodes_path}")

        # 4. Initialise agent
        agent = self.DQNAgent(self)

        # 5. Training loop
        for epoch in range(1, self.n_epochs + 1):
            total_reward = 0.0
            for episode in self.train_episodes:
                for t in range(len(episode['states'])):
                    state = episode['states'][t]
                    action = agent.select_action(state)

                    # reward only if the chosen action matches the *real* action taken
                    reward = episode['rewards'][t] if action == episode['actions'][t] else 0.0
                    done = (t == len(episode['states']) - 1)
                    next_state = episode['states'][min(t + 1, len(episode['states']) - 1)]

                    agent.buffer.push(state, action, reward, next_state, done)
                    total_reward += reward

                agent.train_step()

            logging.info(f"Epoch {epoch:2d} | Train reward: {total_reward:+.2f}")

        # 6. Save model + normalisation stats
        # 6. Save model
        save_path = self.cfg.MODEL_SAVE_PATH  # ← from config
        torch.save({
            'policy_net': agent.policy_net.state_dict(),
            'cost_mean': self.cost_mean,
            'cost_std': self.cost_std,
            'open_mean': self.open_mean,
            'open_std': self.open_std,
            'max_actions': self.max_actions,
        }, save_path)
        logging.info(f"Model saved to {save_path}")

        # -----------------------------------------------------------------
        # 7. EVALUATION AFTER TRAINING
        # -----------------------------------------------------------------
        logging.info("=== Starting Evaluation on Test Set ===")
        total_reward, accuracy = self.predict(agent, self.test_episodes)
        logging.info(
            f"Evaluation Complete → "
            f"Total Reward: {total_reward:.3f}, "
            f"Same courier as previous: {accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------------------- #
    # 9. predict()
    # --------------------------------------------------------------------- #
    def predict(self, agent, test_episodes):
        agent.policy_net.eval()
        total_reward = 0.0
        correct_actions = 0
        total_actions = 0

        with torch.no_grad():
            for episode in test_episodes:
                for t in range(len(episode['states'])):
                    state = torch.FloatTensor(
                        episode['states'][t]).unsqueeze(0).to(self.device)
                    q_values = agent.policy_net(state)
                    action = q_values.argmax().item()

                    true_action = episode['actions'][t]
                    reward = episode['rewards'][
                        t] if action == true_action else 0

                    total_reward += reward
                    if action == true_action:
                        correct_actions += 1
                    total_actions += 1

        avg_reward = total_reward / len(
            test_episodes) if test_episodes else 0
        accuracy = correct_actions / total_actions if total_actions > 0 else 0

        print(f"Evaluation Results:")
        print(f"   Total Reward: {total_reward:.3f}")
        print(f"   Avg Reward per Episode: {avg_reward:.3f}")
        print(
            f"   Same Courier Selection as Previous: {accuracy * 100:.2f}% "
            f"({correct_actions}/{total_actions})")

        return total_reward, accuracy

