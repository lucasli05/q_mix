import torch
from torch import nn, optim
import torch.nn.functional as F

from .utils import *
from .model import *

# 训练函数
def QMIXtrain(replay_buffer_good, replay_buffer_bad,model, target_model, gamma, lr = 1e-3, batch_size = 64):
    idxs_good ,is_weights_good,id_ep_good, s_ep_good, a_ep_good, a_onehot_ep_good, r_ep_good, s_next_ep_good, done_ep_good, obs_ep_good, obs_next_ep_good, a_pre_ep_good, a_pre_onehot_ep_good, action_mask_ep_good, loss_mask_ep_good= replay_buffer_good.sample(int(batch_size/2))
    episode_batch_good  =id_ep_good, s_ep_good, a_ep_good, a_onehot_ep_good, r_ep_good, s_next_ep_good, done_ep_good, obs_ep_good, obs_next_ep_good, a_pre_ep_good, a_pre_onehot_ep_good, action_mask_ep_good, loss_mask_ep_good
    idxs_bad ,is_weights_bad,id_ep_bad, s_ep_bad, a_ep_bad, a_onehot_ep_bad, r_ep_bad, s_next_ep_bad, done_ep_bad, obs_ep_bad, obs_next_ep_bad, a_pre_ep_bad, a_pre_onehot_ep_bad, action_mask_ep_bad, loss_mask_ep_bad= replay_buffer_bad.sample(int(batch_size/2))
    episode_batch_bad  =id_ep_bad, s_ep_bad, a_ep_bad, a_onehot_ep_bad, r_ep_bad, s_next_ep_bad, done_ep_bad, obs_ep_bad, obs_next_ep_bad, a_pre_ep_bad, a_pre_onehot_ep_bad, action_mask_ep_bad, loss_mask_ep_bad
    episode_batch = id_ep_good+id_ep_bad,s_ep_good+s_ep_bad, a_ep_good+ a_ep_bad, a_onehot_ep_good+a_onehot_ep_bad, r_ep_good+r_ep_bad, s_next_ep_good+s_next_ep_bad, done_ep_good+done_ep_bad, obs_ep_good+obs_ep_bad, obs_next_ep_good+obs_next_ep_bad, a_pre_ep_good+a_pre_ep_bad,\
          a_pre_onehot_ep_good+a_pre_onehot_ep_bad, action_mask_ep_good+action_mask_ep_bad, loss_mask_ep_good+loss_mask_ep_bad
    id_ep, s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep, max_steps = trans_to_tensor(episode_batch, model.device)
    hidden_s = torch.zeros(1, batch_size, model.agent_model[0].hidden_size).to(model.device)
    q_val_ls = []
    q_target_ls = []
    for i in range(model.num_agent):
        agent_inputs = torch.cat([obs_ep[:, :, i, :], a_pre_onehot_ep[:, :, i, :], id_ep[:, :, i, :]], -1) # batch_size, seq_len, n_agent, obs_size // batch_size, seq_len, n_agent, action_size
        q_idx, _ = model.agent_model[i](agent_inputs, hidden_s, max_step = max_steps)   #q(s)    [batch,step,action_num]
        q_val = torch.gather(q_idx, -1, a_ep[:, :, i].unsqueeze(-1))   #q(s,a)     #[batch , step, 1]
        q_val_ls.append(q_val)  #agent episode q value
        
        agent_next_inputs = torch.cat([obs_next_ep[:, :, i, :], a_onehot_ep[:, :, i, :], id_ep[:, :, i, :]], -1)
        q_target, _ = target_model.agent_model[i](agent_next_inputs, hidden_s, max_step = max_steps)
        q_target[action_mask_ep[:, :, i, :] == 0] = -9999999
        q_target = r_ep + gamma * (torch.max(q_target, -1)[0]).unsqueeze(-1) * (1 - done_ep)
        q_target_ls.append(q_target)
    qval_ls = torch.cat(q_val_ls, -1)
    q_target_ls = torch.cat(q_target_ls, -1)
    q_tot = model(qval_ls, s_ep).view(batch_size, -1, 1) 
    q_target_tot = target_model(q_target_ls, s_next_ep).view(batch_size, -1, 1) .detach()
    loss_step = (((q_target_tot - q_tot)**2) * loss_mask_ep)
    loss_episode =torch.mean(loss_step , dim = 1)
    is_weights = np.concatenate((is_weights_good,is_weights_bad))
    is_weights_tensor = torch.tensor(is_weights)
    is_weights_reshaped = is_weights_tensor.unsqueeze(1)
    is_weights_reshaped = is_weights_reshaped.to("cuda")
    loss_tot = (loss_episode*is_weights_reshaped).mean()
    # loss_tot = (((q_target_tot - q_tot)**2) * loss_mask_ep).mean()
    error_tot = ((q_target_tot - q_tot)* loss_mask_ep)
    for i in range(int(batch_size/2)):
        error_episode =  error_tot[i,:,:]
        error = abs(error_episode.mean().item())
        idx = idxs_good[i]
        replay_buffer_good.update(idx,error)
    for i in range(int(batch_size/2),int(batch_size)):
        error_episode =  error_tot[i,:,:]
        error = abs(error_episode.mean().item())
        idx = idxs_bad[i-int(batch_size/2)]
        replay_buffer_bad.update(idx,error)
    
    # optimize
    model.optimizer.zero_grad()    
    loss_tot.backward()    #计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  #: 对梯度进行裁剪，防止梯度爆炸问题。
    model.optimizer.step()   #更新模型
    return loss_tot,q_tot,q_target_tot,agent_inputs

