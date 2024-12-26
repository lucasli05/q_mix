import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import collections
import random


# Qmix 的 agent 网络


#所有Agent共享一个网络，input_shape = obs_shape + n_agents+n_actions
###  MLP input layer
###  GRU  : 上一回合输出的hidden layer 为当前回合的input
###  MLP output layer

class QMIXAgent_net(nn.Module):
    def __init__(self, args):
        super(QMIXAgent_net, self).__init__()
        self.input_size, self.hidden_size, self.action_size, self.device = args
        self.fc1 = nn.Linear(self.input_size, 64)

        ##input shape for GRU (batch_size, sequence_length, input_size)
        self.gru_net = nn.GRU(64, self.hidden_size, batch_first = True)    ##input_size ,hidden_size,num_layers
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, inputs, hidden_s, max_step = 1):
        fc1_op = self.fc1(inputs)
        fc1_op = fc1_op.view(-1, max_step, 64)  ##batch_at first
        gru_op, hidden_next = self.gru_net(fc1_op, hidden_s)
        #output:(batch_size, sequence_length, hidden_size)
        gru_op = gru_op.view(-1, max_step, self.hidden_size)
        q_val = self.fc2(gru_op).view(-1, max_step, self.action_size)
        return q_val, hidden_next

    def get_action(self, obs, hidden_s, epsilon, action_mask):
        inputs = obs.unsqueeze(0)    # (1, 观测特征大小) 的张量
        q_val, h_s = self(inputs, hidden_s)    ###forward
        q_val = q_val.squeeze(0)
        q_val = q_val.squeeze(0) * torch.FloatTensor(action_mask).to(self.device)
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                q_val[i] = -float('inf')
        seed = np.random.rand()
        if seed > epsilon:    ##epsilon greedy
            print('greedy policy for q',q_val)
            return torch.argmax(q_val).item(), h_s    ##greedy
        else:
            avail_actions_ind = np.nonzero(action_mask)[0]    ##random
            action = np.random.choice(avail_actions_ind)
            return action, h_s

# Qmix 的 mix 网络
class QMIXMixing_net(nn.Module):
    def __init__(self, args):
        super(QMIXMixing_net, self).__init__()
        self.num_agent, self.joint_obs_size, self.obs_info, self.action_info, self.device, self.lr = args
        self.hidden_nums = [64, 1]
        # 超网络 用于生成混合加权各个代理的q值的权重
        self.hyper_netw1 = nn.Linear(self.joint_obs_size, self.num_agent * self.hidden_nums[0])
        self.hyper_netw2 = nn.Linear(self.joint_obs_size, self.hidden_nums[0] * self.hidden_nums[1])
        self.hyper_netb1 = nn.Linear(self.joint_obs_size, self.hidden_nums[0])
        self.hyper_netb2 = nn.Linear(self.joint_obs_size, self.hidden_nums[1])
        self.agent_model = nn.ModuleList([QMIXAgent_net(args = (self.obs_info[i] + self.action_info[i] + self.num_agent, 32, self.action_info[i], self.device)) for i in range(self.num_agent)])
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, q_vals, inputs):
        weights1 = torch.abs(self.hyper_netw1(inputs)).view(-1, self.num_agent, self.hidden_nums[0])   #确保权重是非负的
        weights2 = torch.abs(self.hyper_netw2(inputs)).view(-1, self.hidden_nums[0], self.hidden_nums[1])
        b1 = self.hyper_netb1(inputs).view(-1, 1, self.hidden_nums[0])
        b2 = self.hyper_netb2(inputs).view(-1, 1, self.hidden_nums[1])

        q_vals = q_vals.view(-1, 1, self.num_agent)
        q_tot = torch.bmm(torch.bmm(q_vals, weights1) +  b1, weights2) + b2  #QMIX 算法的混合网络使用了两层线性层进行局部 Q 值的加权汇聚。
        return q_tot


# 记忆库
class Replaybuffer():
    def __init__(self, args):
        self.size = args
        self.mem_list = collections.deque(maxlen = self.size)    #双端队列


        self.mem_good_list = collections.deque(maxlen = self.size)

    @property
    def mem_len(self):
        return len(self.mem_list)
        # return len(self.mem_list)+len(self.mem_good_list)
    @property
    def mem_good_len(self):
        return len(self.mem_good_list)
    
    def save_trans(self, trans):
        self.mem_list.append(trans)
    def save_good_trans(self, trans):
        self.mem_good_list.append(trans)
    
    def sample_batch(self, batch_size = 64):   #用于从缓存中随机采样一个批次的经验样本，以用于训练智能体的神经网络，默认64
        episode_batch = random.sample(self.mem_list, batch_size)
        id_ep, s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep = ([] for _ in range(13))
        for episode in episode_batch:
            id_ls, s_ls, a_ls, a_onehot_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, a_pre_onehot_ls, action_mask_ls, loss_mask_ls = ([] for _ in range(13))
            for trans in episode:
                id_, s, a, a_onehot, r, s_next, done, obs, obs_next, a_pre, a_pre_onehot, action_mask, loss_mask = trans
                id_ls.append(id_)
                s_ls.append(s)
                a_ls.append(a)
                a_onehot_ls.append(a_onehot)
                r_ls.append([r])
                s_next_ls.append(s_next)
                done_ls.append([done])
                obs_ls.append(obs)
                obs_next_ls.append(obs_next)
                a_pre_ls.append(a_pre)
                a_pre_onehot_ls.append(a_pre_onehot)
                action_mask_ls.append(action_mask)
                loss_mask_ls.append([loss_mask])
            id_ep.append(id_ls)  #每个episode中的 id_ls
            s_ep.append(s_ls)
            a_ep.append(a_ls)
            a_onehot_ep.append(a_onehot_ls)
            r_ep.append(r_ls)
            s_next_ep.append(s_next_ls)
            done_ep.append(done_ls)
            obs_ep.append(obs_ls)
            obs_next_ep.append(obs_next_ls)
            a_pre_ep.append(a_pre_ls)
            a_pre_onehot_ep.append(a_pre_onehot_ls)
            action_mask_ep.append(action_mask_ls)
            loss_mask_ep.append(loss_mask_ls)

        return id_ep, s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep





      