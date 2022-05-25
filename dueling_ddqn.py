import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import random

from replay_buffers import BasicBuffer
from model import DuelingDQN
import matplotlib.pyplot as plt

class DuelingAgent:

    #buffer size？
    def __init__(self, env, c_s, learning_rate=0.01, gamma=0.99, buffer_size=10000):
        self.env = env
        self.c_s = c_s
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN(self.c_s, 2).to(self.device)
      
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        state = torch.DoubleTensor(state).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        action_bound=[0,1]
        if(np.random.randn() > eps):
            action = random.sample(action_bound, 1)
            action = action[0]
            return action

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.DoubleTensor(next_states).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)

        return loss

    def update(self, batch_size):
        for i in range(50):
            batch = self.replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)
            #print("update loss ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size
                     ,request_dataset, v2i_rate, v2i_rate_mbs,vehicle_epoch, vehicle_request_num):

    episode_rewards = []

    cache_efficiency_list=[]
    cache_efficiency2_list = []
    request_delay_list=[]

    for episode in range(max_episodes):
        state , _ , _= env.reset()
        episode_reward = 0
        # state 是前100个电影

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, cache_efficiency, cache_efficiency2, request_delay = env.step(action, request_dataset, v2i_rate,v2i_rate_mbs, vehicle_epoch, vehicle_request_num, step)
            agent.replay_buffer.push(state, action, reward, next_state)
            episode_reward += reward

            #if len(agent.replay_buffer) > batch_size:
            if len(agent.replay_buffer) % batch_size == 0:
                agent.update(batch_size)

            if step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                cache_efficiency_list.append(cache_efficiency)
                cache_efficiency2_list.append(cache_efficiency2)
                request_delay_list.append(request_delay)
                break

            state = next_state
        print(len(agent.replay_buffer.buffer))

    # 30个episode的数据
    return episode_rewards, cache_efficiency_list, request_delay_list

def mini_batch_train_episode(env, agent, max_episodes, max_steps, batch_size
                     ,request_dataset, v2i_rate, v2i_rate_mbs,vehicle_epoch, vehicle_request_num):

    episode_rewards = []

    cache_efficiency_list=[]
    cache_efficiency2_list = []
    request_delay_list=[]

    for episode in range(max_episodes):
        state , _ , _= env.reset()
        episode_reward = 0
        # state 是前100个电影

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, cache_efficiency, cache_efficiency2, request_delay = env.step(action, request_dataset, v2i_rate,v2i_rate_mbs, vehicle_epoch, vehicle_request_num, step)
            agent.replay_buffer.push(state, action, reward, next_state)
            episode_reward += reward

            if episode==0:
                if step==0:
                    cache_efficiency_list.append(cache_efficiency)
                    cache_efficiency2_list.append(cache_efficiency2)
                    request_delay_list.append(request_delay)

            #if len(agent.replay_buffer) > batch_size:
            if len(agent.replay_buffer) % batch_size == 0:
                agent.update(batch_size)

            if step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                cache_efficiency_list.append(cache_efficiency)
                cache_efficiency2_list.append(cache_efficiency2)
                request_delay_list.append(request_delay)
                break

            state = next_state
        print(len(agent.replay_buffer.buffer))

    # 30个episode的数据
    return episode_rewards, cache_efficiency_list, request_delay_list

def mini_batch_train_density(env, agent, max_episodes, max_steps, batch_size
                     ,request_dataset, v2i_rate, vehicle_epoch, vehicle_request_num, vehicle_density):

    episode_rewards = []

    cache_efficiency_list=[]
    cache_efficiency2_list = []
    request_delay_list=[]

    for episode in range(max_episodes):
        state , _ , _= env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, cache_efficiency, cache_efficiency2, request_delay = env.step_density(action, request_dataset, v2i_rate, vehicle_epoch, vehicle_request_num, step, vehicle_density)
            agent.replay_buffer.push(state, action, reward, next_state)
            episode_reward += reward

            #if len(agent.replay_buffer) > batch_size:
            if len(agent.replay_buffer) % batch_size == 0:
                agent.update(batch_size)

            if step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                cache_efficiency_list.append(cache_efficiency)
                cache_efficiency2_list.append(cache_efficiency2)
                request_delay_list.append(request_delay)
                break
            state = next_state

    return episode_rewards, cache_efficiency_list, request_delay_list