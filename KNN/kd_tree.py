# @author: JinnTaoo
# @email: jinntaoor@gmail.com


import numpy as np
import math


class KDNode:
    def __init__(self, point=None, label=None, split=None, parent=None, left=None, right=None):
        """
        :param point: data point
        :param split: k-th axis use for split
        :param left: left child
        :param right: right child
        """
        self.point = point
        self.label = label
        self.split = split
        self.parent = parent
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, _list, _label_list):
        self.__length = 0
        self.__root = self.__create_kd_tree(_list, _label_list)

    @property
    def length(self):
        return self.__length

    @property
    def root(self):
        return self.__root

    def __create_kd_tree(self, _list, _label_list, parent_node=None):
        """
        :param _list: shape为(m,n)的列表，表示m个样本
        :param _label_list: 样本的标签
        :param parent_node: 父节点
        :return: kd树的根节点
        """
        data_array = np.array(_list)
        m, n = data_array.shape
        label_array = np.array(_label_list).reshape(m, 1)
        if m == 0:
            return None
        split = np.var(data_array, axis=0, keepdims=False).argmax()
        index_list_of_max_var = data_array[:, split].argsort()
        mid_point_index = index_list_of_max_var[int(m / 2)]
        if m == 1:
            self.__length += 1
            return KDNode(point=data_array[mid_point_index],
                          label=label_array[mid_point_index],
                          split=split,
                          parent=parent_node, left=None, right=None)
        node = KDNode(point=data_array[mid_point_index],
                      label=label_array[mid_point_index],
                      split=split,
                      parent=parent_node)

        left = self.__create_kd_tree(data_array[index_list_of_max_var[: int(m / 2)]],
                                     label_array[index_list_of_max_var[:int(m / 2)]],
                                     node)
        if m == 2:
            right = None
        else:
            right = self.__create_kd_tree(data_array[index_list_of_max_var[int(m / 2) + 1:]],
                                          label_array[index_list_of_max_var[int(m / 2) + 1:]],
                                          node)
        node.left = left
        node.right = right
        self.__length += 1
        return node

    def find_nn(self, query):
        """
        查找最近邻
        :param query: 待查的样本
        :return: 距离最近的样本
        """
        query_array = np.array(query)
        nn = self.__root
        if self.length == 1:
            return nn
        while True:
            cur_split = nn.split
            if query[cur_split] == nn.point[cur_split]:
                return nn
            elif query[cur_split] < nn.point[cur_split]:
                if nn.left is None:
                    return nn
                nn = nn.left
            else:
                if nn.right is None:
                    return nn
                nn = nn.right

    def find_nn_backtracking(self, query):
        """
        先建立查找列表，再回溯查找最近邻
        :param query: 待查询点
        :return: 距点point最近的点，以及其距离
        """
        nn = self.root
        tmp_nn = self.root
        min_dist = calc_dist(query, nn.point)
        visited = list()
        while tmp_nn:
            visited.append(tmp_nn)
            dist_ = calc_dist(query, tmp_nn.point)
            if min_dist > dist_:
                nn = tmp_nn
                min_dist = dist_
            if tmp_nn.split is None:
                break
            split_ = tmp_nn.split
            if query[split_] <= tmp_nn.point[split_]:
                tmp_nn = tmp_nn.left
            else:
                tmp_nn = tmp_nn.right
        # 回溯查找
        while visited:
            back_point = visited.pop()
            split_ = back_point.split
            print("back.point= ", back_point.point)
            # 判断是否需要进入父节点的子空间搜索
            if abs(query[split_] - back_point.point[split_]) < min_dist:
                if query[split_] <= back_point.point[split_]:
                    tmp_nn = back_point.right
                else:
                    tmp_nn = back_point.left
                if tmp_nn:
                    visited.append(tmp_nn)
                    cur_dist = calc_dist(point, tmp_nn.point)
                    if min_dist > cur_dist:
                        min_dist = cur_dist
                        nn = tmp_nn.point
            return nn, min_dist


def create_kdTree(data_mat):
    if len(data_mat) == 0:
        return
    if len(data_mat) == 1:
        return KDNode(point=data_mat[0], split=0)
    split = np.var(data_mat, axis=0, keepdims=False).argmax()  # 计算方差最大的feature所在列
    data_mat = data_mat[data_mat[:, split].argsort()]  # 按上述列排序
    mid = int(len(data_mat) / 2)
    mid_point = data_mat[mid]
    root = KDNode(point=mid_point, split=split)
    root.left = create_kdTree(data_mat[:mid])
    root.right = create_kdTree(data_mat[mid + 1:])
    return root


def find_nearest_neighbour(root, point):
    """
    先建立查找列表，再回溯查找最近邻
    :param root: 树根
    :param point: 待查询点
    :return: 距点point最近的点，以及其距离
    """
    nearest_point = root.point
    min_dist = calc_dist(point, nearest_point)
    visited = list()
    tmp_node = root
    while tmp_node:
        visited.append(tmp_node)
        dist_ = calc_dist(point, tmp_node.point)
        if min_dist > dist_:
            nearest_point = tmp_node.point
            min_dist = dist_
        if tmp_node.split is None:
            break
        split_ = tmp_node.split
        if point[split_] <= tmp_node.point[split_]:
            tmp_node = tmp_node.left
        else:
            tmp_node = tmp_node.right
    # 回溯查找
    while visited:
        back_point = visited.pop()
        split_ = back_point.split
        print("back.point= ", back_point.point)
        # 判断是否需要进入父节点的子空间搜索
        if abs(point[split_] - back_point.point[split_]) < min_dist:
            if point[split_] <= back_point.point[split_]:
                tmp_node = back_point.right
            else:
                tmp_node = back_point.left
            if tmp_node:
                visited.append(tmp_node)
                cur_dist = calc_dist(point, tmp_node.point)
                if min_dist > cur_dist:
                    min_dist = cur_dist
                    nearest_point = tmp_node.point
        return nearest_point, min_dist


def calc_dist(point1, point2):
    """
    计算两点之间的欧式距离
    :param point1:
    :param point2:
    :return:
    """
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))


def calc_variance(data_list):
    """
    or you can use np.std instead
    :param self:
    :param data_list: calc. its' variance
    :return: variance of data in data_list
    """
    mean = sum(data_list) / len(data_list)
    var = sum(map(lambda x: (x - mean) ** 2, data_list)) / len(data_list)
    return var


if __name__ == "__main__":
    x = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    y = [0, 0, 1, 0, 1, 1]
    x_mat = np.array(x)
    tree1 = create_kdTree(x_mat)
    point = [2, 4]
    nn1, dist1 = find_nearest_neighbour(tree1, point)
    tree2 = KDTree(x, y)
    nn2 = tree2.find_nn(point)
    nn2_2, dist2_2 = tree2.find_nn_backtracking(point)

    # kdt = KDTree()
    # var = kdt.calc_variance(data_list=x)
    # print(var)
