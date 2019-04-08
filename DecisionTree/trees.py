# -*- coding: utf-8 -*-
# @Time   : 2019/4/7 20:32
# @Author : JinnTaoo

import math
import operator


def create_dataset():
    dataset = [[1, 2, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def calc_shannon_entropy(dataset):
    """每行为一个样本，最后一位为标签"""
    num_entries = len(dataset)
    label_counts = dict()
    # 为所有可能分类创建字典
    for feature_vec in dataset:
        curr_label = feature_vec[-1]
        if curr_label not in label_counts:
            label_counts[curr_label] = 1
        else:
            label_counts[curr_label] += 1
    shanno_ent = 0.0
    for k in label_counts:
        prob = float(label_counts[k]) / num_entries
        shanno_ent -= prob * math.log(prob, 2)
    return shanno_ent


def split_dataset(dataset, axis, value):
    """
    :param dataset: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征的返回值
    :return:
    """
    ret_dataset = list()
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduced_feat_vec = feature_vec[:axis]
            reduced_feat_vec.extend(feature_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    """利用信息增益选择最好的数据集划分方式"""
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_entropy(dataset)
    best_info_gain, best_feature = 0.0, -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)  # 创建唯一的分类标签集合
        new_entropy = 0.0

        # 计算每种划分方式的信息熵
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy  # 计算信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = dict()
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        else:
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):  # 类别完全相同则停止划分
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)  # 遍历完所有特征时返回出现次数最多的
    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    class_label = None
    for k in second_dict.keys():
        if test_vec[feat_index] == k:
            if type(second_dict[k]).__name__ == 'dict':
                class_label = classify(second_dict[k], feat_labels, test_vec)
            else:
                class_label = second_dict[k]
    return class_label


def store_tree(input_tree, file_name):
    import pickle
    with open(file_name, 'w', encoding='utf-8') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(file_name):
    import pickle
    with open(file_name, 'r', encoding='utf-8') as fr:
        return pickle.load(fr)


if __name__ == "__main__":
    dataset, labels = create_dataset()
    shanno_ent = calc_shannon_entropy(dataset)
    print(shanno_ent)

    # print('\ntest for split_dataset: ')
    # print(split_dataset(dataset, 0, 1))
    # print(split_dataset(dataset, 0, 0))
    # print()

    # print('\nchoose_best_feature_to_split: ')
    # ind = choose_best_feature_to_split(dataset)
    # print('axis = ', ind)

    print('\ncreate_tree: ')
    my_tree = create_tree(dataset, labels)
    print('my_tree: \n', my_tree)

    print('\nclassify: ')
    dataset, labels = create_dataset()
    print(classify(my_tree, labels, [1, 0]))
    print(classify(my_tree, labels, [1, 1]))
