'''
    This is a benchmark to see how DDPG works with different GNN for LDO
'''

import torch
import numpy as np
import os
import gymnasium as gym

import pickle  

from ckt_graphs import GraphLDOtestbench
from ddpg import DDPGAgent
from datetime import datetime

from utils import ActionNormalizer, OutputParser
from models import ActorCriticRGCN, ActorCriticGCN, ActorCriticGAT, ActorCriticMLP

from LDO_TB import LDOtestbenchEnv

date = datetime.today().strftime('%Y-%m-%d')
PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CktGraph = GraphLDOtestbench
GNN = ActorCriticRGCN # you can select other GNN
rew_eng = CktGraph().rew_eng

""" Run intial op experiment """

run_intial = False
if run_intial == True:
    env = LDOtestbenchEnv()
    env._init_random_sim(100)
    
""" Regsiter the environemnt to gymnasium """
from gymnasium.envs.registration import register

env_id = 'sky130LDO_TB-v0'
env_dict = gym.envs.registration.registry.copy()

for env in env_dict:
    if env_id in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry[env]

print("Register the environment")
register(
        id = env_id,
        entry_point = 'LDO_TB:LDOtestbenchEnv',
        max_episode_steps = 50,
        )
env = gym.make(env_id)  


# parameters
num_steps = 10000
memory_size = 100000
batch_size = 128
noise_sigma = 2 # noise volume
noise_sigma_min = 0.1
noise_sigma_decay = 0.9995 # if 1 means no decay
initial_random_steps = 2000
noise_type = 'uniform' 

agent = DDPGAgent(
    env, 
    CktGraph(),
    GNN().Actor(CktGraph()),
    GNN().Critic(CktGraph()),
    memory_size, 
    batch_size,
    noise_sigma,
    noise_sigma_min,
    noise_sigma_decay,
    initial_random_steps=initial_random_steps,
    noise_type=noise_type, 
)

# train the agent
agent.train(num_steps)

""" Replay the best results """
memory = agent.memory
rews_buf = memory.rews_buf[:num_steps]
info  = memory.info_buf[:num_steps]
best_design = np.argmax(rews_buf)
best_action = memory.acts_buf[best_design]
best_reward = np.max(rews_buf)
agent.env.step(best_action) # run the simulations

results = OutputParser(CktGraph())
op_results = results.dcop('LDO_TB_op')

# saved agent's actor and critic network, memory buffer, and agent
save = True
if save == True:
    model_weight_actor = agent.actor.state_dict()
    save_name_actor = f"Actor_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pth"
      
    model_weight_critic = agent.critic.state_dict()
    save_name_critic = f"Critic_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pth"
      
    torch.save(model_weight_actor, PWD + "/saved_weights/" + save_name_actor)
    torch.save(model_weight_critic, PWD + "/saved_weights/" + save_name_critic)
    print("Actor and Critic weights have been saved!")

    # save memory
    with open(f'./saved_memories/memory_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pkl', 'wb') as memory_file:
        pickle.dump(memory, memory_file)

    np.save(f'./saved_memories/rews_buf_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}', rews_buf)

    # save agent
    with open(f'./saved_agents/DDPGAgent_{CktGraph().__class__.__name__}_{date}_noise={noise_type}_reward={best_reward:.2f}_{GNN().__class__.__name__}_rew_eng={rew_eng}.pkl', 'wb') as agent_file:
        pickle.dump(agent, agent_file)
