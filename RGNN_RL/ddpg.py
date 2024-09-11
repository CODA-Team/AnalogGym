import numpy as np

from typing import List, Tuple, Dict
from copy import deepcopy
import torch
from torch.nn import LazyLinear
import torch.nn.functional as F
import torch.optim as optim

from utils import trunc_normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

class ReplayBuffer:
    """A simple numpy replay buffer."""
    def __init__(self, CktGraph, size: int, batch_size: int = 32):  

        self.num_node_features = CktGraph.num_node_features 
        self.action_dim = CktGraph.action_dim
        self.num_nodes = CktGraph.num_nodes
        
        """Initializate."""
        self.obs_buf = np.zeros(
            [size, self.num_nodes, self.num_node_features], dtype=np.float32)  
        self.next_obs_buf = np.zeros(
            [size, self.num_nodes, self.num_node_features], dtype=np.float32)  
        self.acts_buf = np.zeros([size, self.action_dim], dtype=np.float32)    
        self.rews_buf = np.zeros([size], dtype=np.float32)                     
        self.done_buf = np.zeros([size], dtype=np.float32)                    
        self.info_buf = np.zeros([size], dtype=object)# store the performance in each step                        
        self.max_size, self.batch_size = size, batch_size                      
        self.ptr, self.size, = 0, 0                                           

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs                 
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.info_buf[self.ptr] = info                 
        self.ptr = (self.ptr + 1) % self.max_size     
        self.size = min(self.size + 1, self.max_size) 
    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)  
        return dict(obs=self.obs_buf[idxs],                                      
                    next_obs=self.next_obs_buf[idxs],                            
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int: 
        return self.size


class DDPGAgent:
    """DDPGAgent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise: noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(

        self,
        env,
        CktGraph,
        Actor,
        Critic,
        memory_size: int,
        batch_size: int,
        noise_sigma: float,
        noise_sigma_min: float,
        noise_sigma_decay: float,
        noise_type: str,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
    ):
        super().__init__()
        """Initialize."""
        self.noise_sigma = noise_sigma
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_decay = noise_sigma_decay
        self.action_dim = CktGraph.action_dim
        self.env = env
        self.memory = ReplayBuffer(CktGraph, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        self.episode = 0
        self.device = CktGraph.device
        print(self.device)
        self.actor = Actor.to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=3e-4, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=3e-4, weight_decay=1e-4)
        self.transition = list()
        self.total_step = 0

        self.noise_type = noise_type
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:

        """Select an action from the input state."""
        if self.is_test == False: 
            if self.total_step < self.initial_random_steps: # if initial random action should be conducted
                print('*** Random actions ***')
                # in (-1, 1)
                selected_action = np.random.uniform(-1, 1, self.action_dim)
            else:
                print(f'*** Actions with Noise sigma = {self.noise_sigma} ***')

                selected_action = self.actor(
                    torch.FloatTensor(state).to(self.device)
                ).detach().cpu().numpy()  # in (-1, 1)
                selected_action = selected_action.flatten()
                if self.noise_type == 'uniform':
                    print(""" uniform distribution noise """)
                    selected_action = np.random.uniform(np.clip(
                        selected_action-self.noise_sigma, -1, 1), np.clip(selected_action+self.noise_sigma, -1, 1))

                if self.noise_type == 'truncnorm':
                    print(""" truncated normal distribution noise """)
                    selected_action = trunc_normal(selected_action, self.noise_sigma)
                    selected_action = np.clip(selected_action, -1, 1)
                
                self.noise_sigma = max(
                    self.noise_sigma_min, self.noise_sigma*self.noise_sigma_decay)

        else:   
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()  # in (-1, 1)
            selected_action = selected_action.flatten()

        print(f'selected action: {selected_action}')
        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)  

        if self.is_test == False:
            self.transition += [reward, next_state, terminated, info]
            self.memory.store(*self.transition)

        return next_state, reward, terminated, truncated, info

    def update_model(self) -> torch.Tensor:
        print("*** Update the model by gradient descent. ***")
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        samples = self.memory.sample_batch()                                 
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
       
        masks = 1 - done                                         
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def train(self, num_steps: int, plotting_interval: int = 20000):
        """Train the agent."""
        self.is_test = False           

        state, info = self.env.reset() 
        actor_losses = []              
        critic_losses = []
        scores = []
        score = 0

        for self.total_step in range(1, num_steps + 1): 
            print(f'*** Step: {self.total_step} | Episode: {self.episode} ***') 
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, info = self.step(action) 
            
            state = next_state
            score += reward
            if terminated or truncated:
                state, info = self.env.reset()
                self.episode = self.episode + 1
                scores.append(score)
                score = 0

            # if training is ready
            if (
                len(self.memory) >= self.batch_size
                and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step,
                    scores,
                    actor_losses,
                    critic_losses,
                )

        self.env.close()    

    def test(self):
        """Test the agent."""
        self.is_test = True
        state, info = self.env.reset()
        truncated = False
        terminated = False
        score = 0

        performance_list = []
        while truncated == False or terminated == False:    
            action = self.select_action(state)             
            next_state, reward, terminated, truncated, info = self.step(action)
            performance_list.append([action, info])         

            state = next_state
            score += reward
        print(f"score: {score}")        
        print(f"info: {info}")
        self.env.close()

        return performance_list   

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau      
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
        self,
        step: int,
        scores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"step {step}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        clear_output(True)        
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()
