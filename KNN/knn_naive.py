# @author: JinnTaoo
# @email: jinntaoor@gmail.com

import math


def get_knn_naive(points, point, k, dist_func, return_distances=True):
    """
    在points中按照dist_func度量方法查找距point最近的k个点
    :param points: 数据集列表shape：(m, n)
    :param point: 待查点
    :param k: 近邻数
    :param dist_func:距离度量方法
    :param return_distances: 是否返回距离
    :return: k个近邻（及其距离）
    """
    neighbors = list()
    for each_point in points:
        neighbors.append([each_point, dist_func(each_point, point)])
    sorted(neighbors, key=(lambda x: x[1]), reverse=False)
    if return_distances:
        return neighbors[:k]
    else:
        return neighbors[:k, :1]


def Euclid_dist(p1, p2):
    """ 两点的列表 """
    dist = math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(p1, p2)))
    return dist


if __name__ == "__main__":
    points = [[2, 3], [5, 4], [9.6], [4, 7], [8.1], [7, 2]]
    point = [2, 5]
    ret = get_knn_naive(points, point, 2, Euclid_dist, True)
    print(ret)
    print('hello')
