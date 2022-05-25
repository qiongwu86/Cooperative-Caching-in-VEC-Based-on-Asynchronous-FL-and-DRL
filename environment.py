import collections
import numpy as np
import torch
import pandas as pd
import random
from local_update import cache_hit_ratio, cache_hit_ratio2
import math

class CacheEnv(object):

    def __init__(self, popular_content,cache_size):
        self.cache_size=cache_size
        self.popular_content = popular_content       # 推荐的电影
        self.action_bound = [0,1]
        self.reward = np.zeros(shape=1, dtype=float)

        #cache list
        if len(self.popular_content) < self.cache_size:
            self.state = self.popular_content

        if len(self.popular_content) >= self.cache_size:
            self.state = random.sample(list(self.popular_content), self.cache_size) # 状态是随机采样的100个推荐电影

        state1 = []
        for i in range(len(self.popular_content)):
            # 按照内容流行度进行排序
            if self.popular_content[i] in self.state:
                state1.append(self.popular_content[i])
        self.state = state1

        self.last_content=[]        # 剩下的电影
        for i in range(len(self.popular_content)):
            if self.popular_content[i] not in self.state:
                self.last_content.append(self.popular_content[i])
        print('self.last_content',len(self.last_content))
        print('self.cache_size',self.cache_size)

        if len(self.last_content)<self.cache_size:
            self.state2 = []
            for i in range(len(self.last_content)):
                if self.last_content[i] not in self.state:
                    self.state2.append(self.last_content[i])

        if len(self.last_content)>=self.cache_size:
            self.state2 = random.sample(list(self.last_content), self.cache_size)

        # 2个RSU分别存放推荐的电影,第一个放前100个，后一个放后100个
        self.init_state = self.state.copy()
        self.init_cash2 = self.state2.copy()
        self.init_last_content = self.last_content.copy()

    def step(self, action, request_dataset, v2i_rate,v2i_rate_mbs, vehicle_epoch, vehicle_request_num, print_step):
        action = np.clip(action, *self.action_bound)

        if action == 1:

            if len((self.last_content))>=5:
                replace_content = random.sample(list(self.last_content), 5)
                count = 0
                if count < 5:
                    self.state[-count - 1] = replace_content[count]
                    count += 1
            else:
                replace_content = self.last_content
            count = 0
            if count < 5:
                self.state[-count-1]=replace_content[count]
                count+=1

            state1 = []
            for i in range(len(self.popular_content)):
                # 按照内容流行度进行排序
                if self.popular_content[i] in self.state:
                    state1.append(self.popular_content[i])
            self.state = state1

            last_content=[]
            for i in range(len(self.popular_content)):
                if self.popular_content[i] not in self.state:
                    last_content.append(self.popular_content[i])
            self.last_content=last_content

            if len(self.last_content)<=self.cache_size:
                self.state2 = self.last_content
            if len(self.last_content)>self.cache_size:
                self.state2 = random.sample(list(self.last_content), self.cache_size)

        all_vehicle_request_num = 0
        for i in range(len(vehicle_epoch)):
            all_vehicle_request_num += vehicle_request_num[vehicle_epoch[i]]
        #print('=================================all_vehicle_request_num', all_vehicle_request_num,'================================')
        cache_efficiency = cache_hit_ratio(request_dataset, self.state,
                                           all_vehicle_request_num)
        cache_efficiency2 = cache_hit_ratio2(request_dataset, self.state2 , self.state,
                                           all_vehicle_request_num)
        cache_efficiency = cache_efficiency/100
        cache_efficiency2 = cache_efficiency2/100

        reward=0
        request_delay=0
        for i in range(len(vehicle_epoch)):
            vehicle_idx=vehicle_epoch[i]
            reward += cache_efficiency * math.exp(-0.0001 * 8000000 / v2i_rate[vehicle_idx]) * vehicle_request_num[vehicle_idx]
            reward += cache_efficiency2 * math.exp(-0.0001 * 8000000 / v2i_rate[vehicle_idx]
                                                    -0.4 * 8000000 / 15000000) * vehicle_request_num[vehicle_idx]
            reward += (1-cache_efficiency-cache_efficiency2)\
                                        * math.exp(- 0.5999 * 8000000 / (v2i_rate[vehicle_idx]/2))* vehicle_request_num[vehicle_idx]

            request_delay += cache_efficiency * vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]*800


            #print(i,'local rsu delay', vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]*100000)
            request_delay += cache_efficiency2 * (
                    vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]+vehicle_request_num[vehicle_idx] / 15000000) *800
            #print(i,'neighbouring rsu delay',(vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]+vehicle_request_num[vehicle_idx] / 15000000) *100000)
            request_delay +=(1-cache_efficiency-cache_efficiency2)*(vehicle_request_num[vehicle_idx] / (v2i_rate[vehicle_idx]/2))*800

            #print(i,'mbs delay',(vehicle_request_num[vehicle_idx] / v2i_rate_mbs[vehicle_idx]) *100000)
        request_delay = request_delay/len(vehicle_epoch)*1000

        if print_step % 50 ==0:
            print("---------------------------------------------")
            print('all_vehicle_request_num', all_vehicle_request_num)
            print('step:{} RSU1 cache_efficiency:{}'.format(print_step,cache_efficiency))
            print('step:{} RSU2 cache_efficiency:{}'.format(print_step,cache_efficiency2))
            print('step',print_step,'request delay:%f' %(request_delay))
            print("---------------------------------------------")
        return self.state, reward, cache_efficiency, cache_efficiency2, request_delay

    def reset(self):
        return self.init_state, self.init_cash2, self.init_last_content

class CacheEnv_density(object):

    def __init__(self, popular_content, cache_size):
        self.cache_size = cache_size
        self.popular_content = popular_content
        self.action_bound = [0, 1]
        self.reward = np.zeros(shape=1, dtype=float)

        # cache list
        if len(self.popular_content)<self.cache_size:
            self.state = self.popular_content

        if len(self.popular_content) >= self.cache_size:
            self.state = random.sample(list(self.popular_content), self.cache_size)

        state1 = []
        for i in range(len(self.popular_content)):
            # 按照内容流行度进行排序
            if self.popular_content[i] in self.state:
                state1.append(self.popular_content[i])
        self.state = state1

        self.last_content = []
        for i in range(len(self.popular_content)):
            if self.popular_content[i] not in self.state:
                self.last_content.append(self.popular_content[i])
        print('self.last_content', len(self.last_content))
        print('self.cache_size', self.cache_size)

        if len(self.last_content)<self.cache_size:
            self.state2 = []
            for i in range(len(self.last_content)):
                if self.last_content[i] not in self.state:
                    self.state2.append(self.last_content[i])

        if len(self.last_content)>=self.cache_size:
            self.state2 = random.sample(list(self.last_content), self.cache_size)

        self.init_state = self.state.copy()
        self.init_cash2 = self.state2.copy()
        self.init_last_content = self.last_content.copy()

    def step_density(self, action, request_dataset, v2i_rate, vehicle_epoch, vehicle_request_num, print_step, vehicle_density):

        action = np.clip(action, *self.action_bound)

        if action == 1:
            if len((self.last_content))>=5:
                replace_content = random.sample(list(self.last_content), 5)
                count = 0
                if count < 5:
                    self.state[-count - 1] = replace_content[count]
                    count += 1
            else:
                replace_content = self.last_content

            state1 = []
            for i in range(len(self.popular_content)):
                # 按照内容流行度进行排序
                if self.popular_content[i] in self.state:
                    state1.append(self.popular_content[i])
            self.state = state1

            last_content = []
            for i in range(len(self.popular_content)):
                if self.popular_content[i] not in self.state:
                    last_content.append(self.popular_content[i])
            self.last_content = last_content

            if len(self.last_content) < self.cache_size:
                self.state2 = []
                for i in range(len(self.last_content)):
                    if self.last_content[i] not in self.state:
                        self.state2.append(self.last_content[i])

            if len(self.last_content) >= self.cache_size:
                self.state2 = random.sample(list(self.last_content), self.cache_size)

        all_vehicle_request_num = 0
        for i in range(len(vehicle_request_num)):
            all_vehicle_request_num += vehicle_request_num[i]
        #print('len(vehicle_request_num)',len(vehicle_request_num))


        cache_efficiency = cache_hit_ratio(request_dataset, self.state,
                                           all_vehicle_request_num)
        cache_efficiency2 = cache_hit_ratio2(request_dataset, self.state2, self.state,
                                             all_vehicle_request_num)
        cache_efficiency = cache_efficiency / 100
        cache_efficiency2 = cache_efficiency2 / 100

        reward = 0
        request_delay = 0

        for i in range(30):
            vehicle_idx = i
            reward += cache_efficiency * math.exp(-0.0001 * 8000000 / v2i_rate[vehicle_idx]) * vehicle_request_num[vehicle_idx]
            reward += cache_efficiency2 * math.exp(-0.0001 * 8000000 / v2i_rate[vehicle_idx]
                                                    -0.4 * 8000000 / 15000000) * vehicle_request_num[vehicle_idx]
            reward += (1-cache_efficiency-cache_efficiency2)\
                                        * math.exp(- 0.5999 * 8000000 / (v2i_rate[vehicle_idx]/2))* vehicle_request_num[vehicle_idx]

            request_delay += cache_efficiency * vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]*800


            #print(i,'local rsu delay', vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]*100000)
            request_delay += cache_efficiency2 * (
                    vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]+vehicle_request_num[vehicle_idx] / 15000000) *800
            #print(i,'neighbouring rsu delay',(vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx]+vehicle_request_num[vehicle_idx] / 15000000) *100000)
            request_delay +=(1-cache_efficiency-cache_efficiency2)*(vehicle_request_num[vehicle_idx] / (v2i_rate[vehicle_idx]/2))*800

            #print(i,'mbs delay',(vehicle_request_num[vehicle_idx] / v2i_rate_mbs[vehicle_idx]) *100000)
        request_delay = request_delay/15*1000

        if print_step % 50 == 0:
            print("---------------------------------------------")
            print('all_vehicle_request_num', all_vehicle_request_num)

            print('step:{} RSU1 cache_efficiency:{}'.format(print_step, cache_efficiency))
            print('step:{} RSU2 cache_efficiency:{}'.format(print_step, cache_efficiency2))
            print('step', print_step, 'request delay:%f' % (request_delay))
            print("---------------------------------------------")
        return self.state, reward, cache_efficiency, cache_efficiency2, request_delay

    def reset(self):

        # cache list
        # state = random.sample(list(self.popular_content), 50)
        # self.state = []
        # self.cache2 = []
        # for i in range(len(self.popular_content)):
        #     if self.popular_content[i] not in state:
        #         self.cache2.append(self.popular_content[i])
        #     # 按照内容流行度进行排序
        #     if self.popular_content[i] in state:
        #         self.state.append(self.popular_content[i])

        return self.init_state, self.init_cash2, self.init_last_content
