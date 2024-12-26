import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity   # 容量--根节点个数
        self.tree = numpy.zeros(2 * capacity - 1)  #总的node个数
        self.data = numpy.zeros(capacity, dtype=object)  #用于存放data的根节点
        self.n_entries = 0  #目前存放的数据量

    # update to the root node
    def _propagate(self, idx, change):   ##递归更新到root
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):  #递归找到叶子节点
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1    #根节点在tree中的index

        self.data[self.write] = data   #write表示在data中的索引
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p   #更新tree中的p
        self._propagate(idx, change)  #更新上面的p

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])