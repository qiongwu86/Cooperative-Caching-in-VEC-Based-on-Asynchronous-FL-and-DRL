import numpy as np
import time
import random
import math
from options import args_parser
from select_vehicle import select_vehicle, vehicle_p_v

args = args_parser()
class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.BS_position = [0, 1000, 2000]
        self.shadow_std = 8
        self.Decorrelation_distance = 50

    def get_path_loss(self, position):
        distance=0
        if self.BS_position[0]<position<self.BS_position[1]:
            distance = position - self.BS_position[0]
        if self.BS_position[1]<position<self.BS_position[2]:
            distance = position - self.BS_position[1]
        if position>self.BS_position[2]:
            distance = position - self.BS_position[2]
        #128.1+37.6log10(d)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()


    def get_path_loss_mbs(self, position):
        distance=0
        if self.BS_position[0]<position<self.BS_position[1]:
            distance = position - self.BS_position[0]
        if self.BS_position[1]<position<self.BS_position[2]:
            distance = position - self.BS_position[1]
        if position>self.BS_position[2]:
            distance = position - self.BS_position[2]
        #128.1+37.6log10(d)
        return 128.1 + 37.6 * np.log10(
            math.sqrt((4*distance) ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)

class Environ:

    def __init__(self, n_veh, V2I_min, BW ,BW_MBS):

        self.V2Ichannels = V2Ichannels()

        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2I_channels_abs = []
        self.decorrelation_distance = 50

        self.V2I_min = V2I_min
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.vehAntGain = 3
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)

        #self.n_RB = n_RB
        self.n_Veh = n_veh
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = BW  # bandwidth per RB, 180,000 MHz
        self.bandwidth_mbs=1000000


    def Compute_Performance_Train(self):

        # ------------ Compute Interference --------------------
        #self.platoon_V2I_Interference = np.zeros(self.n_Veh)  # V2I interferences
        self.platoon_V2I_Signal = np.zeros(self.n_Veh)  # V2I signals

        # for i in range(self.n_Veh):
        #     for j in range(self.n_Veh):
        #         if i!=j:
        #             self.platoon_V2I_Interference[i] += \
        #                     10 ** ((30 - self.V2I_channels_with_fastfading[j][0] +
        #                             self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        # print('self.platoon_V2I_Interference',self.platoon_V2I_Interference)

        # computing the signals
        for i in range(self.n_Veh):
            self.platoon_V2I_Signal[i] = 10 ** ((7 - self.V2I_channels_with_fastfading[i][0] +
                                                self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(self.platoon_V2I_Signal, self.sig2))

        self.interplatoon_rate = V2I_Rate * self.bandwidth

        return self.interplatoon_rate

    def Compute_Performance_Train_mobility(self,epoch_vehicle_num):

        # ------------ Compute Interference --------------------
        #self.platoon_V2I_Interference = np.zeros(self.n_Veh)  # V2I interferences
        self.platoon_V2I_Signal = np.zeros(epoch_vehicle_num)  # V2I signals
        self.platoon_V2I_Signal_mbs = np.zeros(epoch_vehicle_num)  # V2I signals

        #print('self.V2I_channels_with_fastfading',self.V2I_channels_with_fastfading)

        # computing the signals
        for i in range(epoch_vehicle_num):
            self.platoon_V2I_Signal[i] = 10 ** ((30 - self.V2I_channels_with_fastfading[i][0] +
                                                self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        for i in range(epoch_vehicle_num):
            self.platoon_V2I_Signal_mbs[i] = 10 ** ((20 - self.V2I_channels_with_fastfading_mbs[i][0] +
                                                 self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(self.platoon_V2I_Signal, self.sig2))
        V2I_Rate_mbs = np.log2(1 + np.divide(self.platoon_V2I_Signal_mbs, self.sig2))

        self.interplatoon_rate = V2I_Rate * self.bandwidth
        self.interplatoon_rate_mbs = V2I_Rate_mbs * self.bandwidth_mbs

        return self.interplatoon_rate,self.interplatoon_rate_mbs

    def renew_channel(self, number_vehicle,veh_dis,veh_speed):
        """ Renew slow fading channel """
        self.V2I_Shadowing = np.random.normal(0, 4, number_vehicle)
        self.delta_distance = np.asarray([c * self.time_slow for c in veh_speed])
        self.V2I_pathloss = np.zeros((number_vehicle))
        self.V2I_pathloss_mbs = np.zeros((number_vehicle))
        self.V2I_channels_abs = np.zeros((number_vehicle))
        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(number_vehicle):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(veh_dis[i])
            self.V2I_pathloss_mbs[i] = self.V2Ichannels.get_path_loss_mbs(veh_dis[i])

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing
        self.V2I_channels_abs_mbs = self.V2I_pathloss_mbs + self.V2I_Shadowing

    def renew_channels_fastfading(self):

        """ Renew fast fading channel """
        V2I_channels_with_fastfading = self.V2I_channels_abs[:, np.newaxis]
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading_mbs = self.V2I_channels_abs_mbs[:, np.newaxis]
        self.V2I_channels_with_fastfading_mbs = V2I_channels_with_fastfading_mbs - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading_mbs.shape) +
                   1j * np.random.normal(0, 1, V2I_channels_with_fastfading_mbs.shape)) / math.sqrt(2))

    def new_random_game(self, veh_dis,veh_speed):
        # make a new game
        self.renew_channel(int(self.n_Veh),veh_dis,veh_speed)
        self.renew_channels_fastfading()


def get_state(env, idx):
    """ Get state from the environment """

    V2I_abs = (env.V2I_channels_abs[idx] - 60) / 60.0
    V2I_fast = (env.V2I_channels_with_fastfading[idx, :] - env.V2I_channels_abs[idx] + 10) / 35
    Interference = (-env.Interference_all[idx] - 60) / 60
    return np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(Interference, -1)


if __name__ == '__main__':
    all_pos_weight, veh_speed ,veh_dis= select_vehicle()
    time_slow = 0.1
    delta_distance=[]
    for i in range(len(veh_dis)):
        delta_distance.append(veh_speed[i] * time_slow)
    #c-v2x simulation parameters:
    n_RB = 3  # number of resource blocks
    V2I_min = 300  # minimum required data rate for V2I Communication
    bandwidth = int(180000)
    V2I_Shadowing = np.random.normal(0, 8, args.clients_num)
    env=Environ(args.clients_num, V2I_min, bandwidth, delta_distance, V2I_Shadowing)
    env.new_random_game(veh_dis)  # initialize parameters in env