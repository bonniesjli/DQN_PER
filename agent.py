import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(3e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # History vector (non_episodic)
        self.hist = None
        self.alpha_end = 5e-4

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        # Learn every UPDATE_EVERY time steps.
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        states = torch.from_numpy(states).float().to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(states)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.asarray([[np.argmax(action_value.cpu().data.numpy())] for action_value in action_values])
        else:
            return (np.column_stack([np.random.randint(0, self.action_size, size=(self.num_agents))]))
        
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Get max predicted Q values (for next states) from target model
        next_action_values = self.qnetwork_target(next_states)
        Q_targets_next = next_action_values.detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        action_values = self.qnetwork_local(states)
        Q_expected = action_values.gather(1, actions)

        # Compute loss
        loss = (Q_expected - Q_targets).pow(2)
        prios = loss + 1e-5        
        loss = (loss * weights.unsqueeze(1)).mean()
        # loss = F.mse_loss(Q_expected, Q_targets)
        # print (loss)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update replay buffer ------------------- #
        self.memory.update_priorities(indices, prios.data.cpu().numpy())  
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        # self.memory = deque(maxlen=buffer_size)  
        self.buffer_size = buffer_size
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        self.prob_alpha = 0.6
        self.position = 0
        self.priorities = np.zeros((buffer_size, ), dtype = np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if self.__len__() < self.buffer_size:
            self.memory.append(e)
        if self.__len__() == self.buffer_size:
            self.memory[self.position] = e
            
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, beta = 0.4):
        """Sample a batch of experiences from memory accordingly to priorities."""
        # experiences = random.sample(self.memory, k=self.batch_size)
        
        if self.__len__() == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), self.batch_size, p = probs)
        experiences = [self.memory[idx] for idx in indices]
        
        total = self.__len__()
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype = np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        weights = torch.from_numpy(weights).float().to(device)
  
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, batch_indices, batch_priorities):
            for idx, prio in zip(batch_indices, batch_priorities):
                self.priorities[idx] = prio
                
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
        
        
        