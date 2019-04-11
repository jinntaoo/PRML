# -*- coding: utf-8 -*-
"""
  @Time   : 2019/4/11 9:20
  @Author : JinnTaoo
"""

import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt


def distance(pa, pb, method='euclidian'):
    dist = None
    if method == 'euclidian':
        dist = math.sqrt(sum([(ta - tb) ** 2 for ta, tb in zip(pa, pb)]))
    return dist


def generate_k_points(dataset, k):
    """generate k points randomly"""
    dim = len(dataset[0])
    _min_list, _max_list = list(), list()
    for d in range(dim):
        _min, _max = dataset[0][d], dataset[0][d]
        for p_ind in range(len(dataset)):
            if dataset[p_ind][d] < _min:
                _min = dataset[p_ind][d]
            if dataset[p_ind][d] > _max:
                _max_ind = dataset[p_ind][d]
        _min_list.append(_min)
        _max_list.append(_max)
    # _min_list, _max_list = min(dataset), max(dataset)
    return [[random.uniform(x, y) for x, y in zip(_min_list, _max_list)] for _ in range(k)]


def find_center_point(points):
    """

    :param points: list which contains n point within m feature
    :return: center point of points
    """
    return [each_sum_lin / len(points) for each_sum_lin in map(sum, zip(*points))]


def find_and_assign_to_center(dataset, centers):
    """
    :param dataset:
    :param centers: [[center0], [center1]...]
    :return: {centers_ind1:[p1,p2], centers_ind2:[p3,p4]...}
    """
    # clu_dict = dict.fromkeys(centers, None)
    clu_list = list()
    clu_dict = defaultdict(list)
    for i, point in enumerate(dataset):
        dist = [distance(point, center) for center in centers]
        # clu_list.append([dist.index(min(dist)), i])
        clu_dict[dist.index(min(dist))].append(i)
    return clu_dict


def update_centers(dataset, old_clu_dict):
    """
    update center of each group
    :param dataset: 2d list
    :param old_clu_dict: {center1:[p1,p2], center2:[p3,p4]}, p is the index of dataset
    :return: new k centers of k groups
    """
    ret_centers = []
    for val in old_clu_dict.values():
        _data = [dataset[i] for i in val]
        ret_centers.append(find_center_point(_data))
    return ret_centers


def is_changed(dic1, dic2):
    _is = False
    if dic2 is None or dic1.keys() != dic2.keys():
        return True
    for k in dic1.keys():
        if list(dic1[k]) != list(dic2[k]):
            _is = True
    if _is is False:
        print('xxxx')
    return _is


def k_means(dataset, k):
    k_points = generate_k_points(dataset, k=k)
    clus_dict = find_and_assign_to_center(dataset, k_points)
    old_clus_dict = None
    count = 0
    new_centers = None
    while is_changed(clus_dict, old_clus_dict):
        new_centers = update_centers(dataset, clus_dict)
        old_clus_dict = clus_dict
        clus_dict = find_and_assign_to_center(dataset, new_centers)
        print(count)
        count += 1
    centers = new_centers
    return clus_dict, centers


def show_cluster(dataset, k, cluster_dict, certroids):
    cluster_assment = list()
    for k in cluster_dict.keys():
        for v in cluster_dict[k]:
            cluster_assment.append([v, k])
    ass_dic = dict()
    for it in cluster_assment:
        ass_dic.setdefault(it[0], it[1])
    num_sample, dim = len(dataset), len(dataset[0])
    mark = ['or', 'Dk', '^r', 'sb', 'og', '<r', 'dr', 'pr', '+r', 'og', 'ok',  ]
    if k > len(mark):
        print("k is too large!")
        return 1
    for i in range(num_sample):
        mark_index = int(ass_dic[i])
        # plt.plot(dataset[i][0], dataset[i][1], mark[0])
        plt.plot(dataset[i][0], dataset[i][1], mark[mark_index])
    for i in range(k):
        plt.plot(centers[i][0], centers[i][1], mark[i], markersize=14)

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with open('./test_data.txt', 'r', encoding='utf-8') as fr:
        dataset = [each.split('\t') for each in fr.read().strip().split('\n')]
        for i in range(len(dataset)):
            for j in range(2):
                dataset[i][j] = float(dataset[i][j])

    cluster_dict, centers = k_means(dataset, k=4)

    show_cluster(dataset, 4, cluster_dict, centers)
