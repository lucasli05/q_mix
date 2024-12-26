import random
import numpy as np
from Sum_Tree import SumTree


class Memory:  # SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        episode_batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            episode_batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
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

        return idxs ,is_weight , id_ep, s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep

        # return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)