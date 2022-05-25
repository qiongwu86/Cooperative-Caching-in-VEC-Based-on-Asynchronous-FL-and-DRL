import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain
import matplotlib.pyplot as plt
from scipy import stats

from options import args_parser
from dataset_processing import sampling, average_weights,asy_average_weights, sampling_mobility
from user_cluster_recommend import recommend, Oracle_recommend
from local_update import LocalUpdate, cache_hit_ratio,Asy_LocalUpdate, cache_hit_ratio_compare
from model import AutoEncoder
from utils import exp_details, ModelManager, count_top_items
from Thompson_Sampling import thompson_sampling
from data_set import convert
from select_vehicle import select_vehicle, vehicle_p_v, select_vehicle_mobility, vehicle_p_v_mobility, vehicle_p_v_leaving
from cv2x import V2Ichannels, Environ
from dueling_ddqn import DuelingAgent, mini_batch_train
from environment import CacheEnv


if __name__ == '__main__':
    v2i_rate_all = []
    v2i_rate_mbs_all = []
    idx=0
    # 开始时间
    start_time = time.time()
    # args & 输出实验参数
    args = args_parser()
    exp_details(args)
    # gpu or cpu
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load sample users_group_train users_group_test
    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility(args, args.clients_num)
    print('different epoch vehicle request num',vehicle_request_num)

    data_set = np.array(sample)

    # test_dataset & test_dataset_idx
    test_dataset_idxs = []
    for i in range(args.clients_num):
        test_dataset_idxs.append(users_group_test[i])
    test_dataset_idxs = list(chain.from_iterable(test_dataset_idxs))
    test_dataset = data_set[test_dataset_idxs]

    request_dataset = []
    for i in range(args.epochs):
        request_dataset_idxs=[]
        request_dataset_idxs.append(request_content[i])
        request_dataset_idxs = list(chain.from_iterable(request_dataset_idxs))
        request_dataset.append(data_set[request_dataset_idxs])

    all_pos_weight, veh_speed, veh_dis = select_vehicle_mobility(args.clients_num)
    time_slow = 0.1

    # c-v2x simulation parameters:
    V2I_min = 100  # minimum required data rate for V2I Communication
    bandwidth = int(540000)
    bandwidth_mbs = int(1000000)

    env = Environ(args.clients_num, V2I_min, bandwidth, bandwidth_mbs)
    env.new_random_game(veh_dis,veh_speed)  # initialize parameters in env

    # build model
    global_model = AutoEncoder(int(max(data_set[:, 1])), 100)
    V2Ichannels = V2Ichannels()

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    vehicle_model_dict = [[], [], [], [], [], [], [], [], [], [], [], [], []
                          , [], [], [], [], [], [], []]
    for i in range(args.clients_num):
        vehicle_model_dict[i].append(copy.deepcopy(global_model))
    # copy weights
    global_weights = global_model.state_dict()

    # all epoch weights
    w_all_epochs = dict([(k, []) for k in range(args.epochs)])

    # Training loss
    train_loss = []

    # each epoch train time
    each_epoch_time=[]
    each_epoch_time.append(0)

    vehicle_leaving=[]

    v2i_rate_epoch=dict([(k, []) for k in range(args.epochs)])
    v2i_rate_mbs_epoch = dict([(k, []) for k in range(args.epochs)])

    while idx < args.epochs:

        # 开始
        print(f'\n | Global Training Round : {idx + 1} |\n')

        global_model.train()

        #each vehicle local learning rate
        local_lr = args.lr * max(1,np.log(max(1,idx)))

        local_net = copy.deepcopy(vehicle_model_dict[idx % args.clients_num][-1])
        local_net.to(device)

        #v2i rate
        v2i_rate,v2i_rate_mbs=env.Compute_Performance_Train_mobility(args.clients_num)

        v2i_rate_mbs_all.append(v2i_rate_mbs)
        v2i_rate_all.append(v2i_rate)
        print('v2i rate',v2i_rate)
        print('v2i rate mbs', v2i_rate_mbs)

        v2i_rate_epoch[idx]=v2i_rate
        v2i_rate_mbs_epoch[idx] = v2i_rate_mbs

        v2i_rate_weight=v2i_rate/max(v2i_rate)
        print('vehicle position',veh_dis)
        print('vehicle speed', veh_speed)

        print("vehicle ", idx % args.clients_num + 1, " start training for ", args.local_ep,
              " epochs with learning rate ",local_lr)

        if (1000 - veh_dis[idx % args.clients_num]) / veh_speed[idx % args.clients_num] > 2.5:

            epoch_start_time = time.time()
            local_model = Asy_LocalUpdate(args=args, dataset=data_set,
                                          idxs=users_group_train[idx % args.clients_num])

            w, loss, local_net = local_model.update_weights(
                model=local_net, client_idx=idx % args.clients_num + 1, global_round=idx + 1,
                local_learning_rate=local_lr)

            vehicle_model_dict[idx % args.clients_num].append(local_net)
            v_w=vehicle_model_dict[idx % args.clients_num][-1].state_dict()

            #local weight * (position weight + v2i rate weight)
            for key in v_w.keys():
                v_w[key] = v_w[key] * (
                             0.5 * all_pos_weight[idx % args.clients_num] + 0.5 * v2i_rate_weight[idx % args.clients_num])

            vehicle_model_dict[idx % args.clients_num][-1].load_state_dict(v_w)

            #aggeration

            for name, param in vehicle_model_dict[idx % args.clients_num][-1].named_parameters():
                for name2, param2 in vehicle_model_dict[idx % args.clients_num][-2].named_parameters():
                    if name == name2:
                        param.data.copy_(args.update_decay * param2.data + param.data)

            global_w = asy_average_weights(l=vehicle_model_dict[idx % args.clients_num][-1], g=global_model
                                           , l_old=vehicle_model_dict[idx % args.clients_num][-2],vehicle_all_num=args.clients_num)

            epoch_time = time.time() - epoch_start_time
            each_epoch_time.append(epoch_time)
            global_model.load_state_dict(global_w)

            w_all_epochs[idx] = global_w['linear1.weight'].tolist()

        if idx == args.epochs-1:

            ##DDQN
            cache_size=[50,100,150,200,250,300,350,400]
            cache_efficiency_list = []
            cache_efficiency_without_list = []

            TS_recommend_movies = dict([(k, []) for k in cache_size])

            random_cache_efficiency = np.zeros(len(cache_size))
            Oracle_cache_efficiency = np.zeros(len(cache_size))
            Greedy_cache_efficiency = np.zeros(len(cache_size))
            TS_cache_efficiency = np.zeros(len(cache_size))

            # algorithm  parameters
            # m-ε-greedy ε represents the probability to select files randomly from all the files.
            e = 0.1
            request_delay_list = []
            for i in range(len(cache_size)):
                c_s=cache_size[i]
                MAX_EPISODES = 30
                MAX_STEPS = 200
                BATCH_SIZE = 32
                recommend_movies_c500 = []
                Oracle_recommend_movies=[]

                for j in range(args.clients_num):
                    vehicle_seq = j
                    test_dataset_i = data_set[users_group_test[vehicle_seq]]
                    user_movie_i = convert(test_dataset_i, max(sample['movie_id']))
                    recommend_list = recommend(user_movie_i, test_dataset_i, w_all_epochs[idx])
                    recommend_list500 = count_top_items(int(2.5*c_s), recommend_list)
                    recommend_movies_c500.append(list(recommend_list500))

                    Oracle_recommend_movies.append(list(Oracle_recommend(test_dataset_i, c_s)))

                # AFPCC
                recommend_movies_c500 = count_top_items(int(2.5*c_s), recommend_movies_c500)
                env_rl = CacheEnv(recommend_movies_c500, c_s)
                agent = DuelingAgent(env_rl,c_s)
                episode_rewards, cache_efficiency, request_delay = mini_batch_train(env_rl, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE,
                                                   request_dataset[idx]
                                                   , v2i_rate_epoch[idx],v2i_rate_mbs_epoch[idx],
                                                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                                                   vehicle_request_num[idx])
                cache_efficiency_list.append(cache_efficiency[-1]*100)
                cache_efficiency_without_list.append(cache_efficiency[0]*100)
                request_delay_list.append(request_delay[args.epochs - 1])

                # random caching
                random_caching_movies = list(np.random.choice(range(1, max(sample['movie_id']) + 1), c_s, replace=False))
                random_cache_efficiency[i] = cache_hit_ratio_compare(request_dataset[idx], random_caching_movies)

                # Oracle
                Oracle_recommend_movies = count_top_items(c_s, Oracle_recommend_movies)
                Oracle_cache_efficiency[i] = cache_hit_ratio_compare(request_dataset[idx], Oracle_recommend_movies)

                # Thompson Sampling
                TS_recommend_movies[c_s] = thompson_sampling(args, data_set, test_dataset, c_s)
                TS_cache_efficiency[i] = cache_hit_ratio_compare(request_dataset[idx], TS_recommend_movies[c_s])

            # m-e-greedy
            Greedy_cache_efficiency = Oracle_cache_efficiency * (1 - e) + random_cache_efficiency * e

            # Oracle Caching
            # plt.plot(cache_size, Oracle_cache_efficiency, color='blue', linewidth=1.5, linestyle='-', label='Oracle')
            # plt.scatter(cache_size, Oracle_cache_efficiency, s=20, marker='^', color='blue')
            #print('Oracle_cache_efficiency',Oracle_cache_efficiency)
            # MCAF
            print('MCAF',cache_efficiency_list)
            # m-ε-greedy
            print('Greedy_cache_efficiency',Greedy_cache_efficiency)
            # Thompson Sampling
            print('TS_cache_efficiency',TS_cache_efficiency)
            # Random Caching
            print('random_cache_efficiency',random_cache_efficiency)


            vehicle_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            # Greedy_cache_efficiency
            Greedy_cache_efficiency_request_delay_list = []
            for j in range(len(cache_size)):
                request_delay = 0
                for i in range(len(vehicle_array)):
                    vehicle_idx = vehicle_array[i]
                    request_delay += Greedy_cache_efficiency[j] / 100 * vehicle_request_num[idx][
                        vehicle_idx] / v2i_rate_epoch[idx][vehicle_idx] * 800
                    request_delay += random_cache_efficiency[j] / 100 * (
                                vehicle_request_num[idx][vehicle_idx] / v2i_rate_epoch[idx][vehicle_idx]
                                + vehicle_request_num[idx][vehicle_idx] / 15000000) * 800
                    request_delay += (1 - Greedy_cache_efficiency[j] / 100 - random_cache_efficiency[j] / 100) * (
                            vehicle_request_num[idx][vehicle_idx] / (v2i_rate_epoch[idx][vehicle_idx]/2)) * 800
                request_delay = request_delay * 1000 / len(vehicle_array)
                Greedy_cache_efficiency_request_delay_list.append(request_delay)

            #    TS_cache_efficiency
            TS_cache_efficiency_request_delay_list = []
            for j in range(len(cache_size)):
                request_delay = 0
                for i in range(len(vehicle_array)):
                    vehicle_idx = vehicle_array[i]
                    request_delay += TS_cache_efficiency[j] / 100 * vehicle_request_num[idx][
                        vehicle_idx] / v2i_rate_epoch[idx][vehicle_idx] * 800
                    request_delay += random_cache_efficiency[j] / 100 * (
                                vehicle_request_num[idx][vehicle_idx] / v2i_rate_epoch[idx][vehicle_idx]
                                + vehicle_request_num[idx][vehicle_idx] / 15000000) * 800
                    request_delay += (1 - TS_cache_efficiency[j] / 100 - random_cache_efficiency[j] / 100) * (
                            vehicle_request_num[idx][vehicle_idx] / (v2i_rate_epoch[idx][vehicle_idx]/2)) * 800
                request_delay = request_delay * 1000 / len(vehicle_array)
                TS_cache_efficiency_request_delay_list.append(request_delay)

            random_cache_efficiency_request_delay_list = []
            for j in range(len(cache_size)):
                request_delay = 0
                for i in range(len(vehicle_array)):
                    vehicle_idx = vehicle_array[i]
                    request_delay += random_cache_efficiency[j] / 100 * vehicle_request_num[idx][
                        vehicle_idx] / v2i_rate_epoch[idx][vehicle_idx] * 800
                    request_delay += random_cache_efficiency[j] / 100 * (
                                vehicle_request_num[idx][vehicle_idx] / v2i_rate_epoch[idx][vehicle_idx]
                                + vehicle_request_num[idx][vehicle_idx] / 15000000) * 800
                    request_delay += (1 - random_cache_efficiency[j] / 100 - random_cache_efficiency[j] / 100) * (
                            vehicle_request_num[idx][vehicle_idx] / (v2i_rate_epoch[idx][vehicle_idx]/2)) * 800
                request_delay = request_delay * 1000 / len(vehicle_array)
                random_cache_efficiency_request_delay_list.append(request_delay)

            print('MCAF request delay', request_delay_list)
            print('Greedy_cache_efficiency request delay', Greedy_cache_efficiency_request_delay_list)
            print('TS_cache_efficiency request delay', TS_cache_efficiency_request_delay_list)
            print('random_cache_efficiency request delay', random_cache_efficiency_request_delay_list)

        idx += 1
        veh_dis, veh_speed, all_pos_weight = vehicle_p_v_mobility(veh_dis, epoch_time, args.clients_num, idx, args.clients_num)

        env.renew_channel(args.clients_num, veh_dis, veh_speed)  # update channel slow fading
        env.renew_channels_fastfading()  # update channel fast fading

        if idx > args.epochs:
            break

