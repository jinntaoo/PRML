# -*- coding: utf-8 -*-
# @Time   : 2019/4/6 18:55
# @Author : JinnTaoo

# sklearn中有三种不同类型的朴素贝叶斯，高斯分布型（假设属性特征服从正态分布）用于分类和多项式型和伯努利型
# 此处使用20newsgroups文本数据进行文本分类实验
# 不使用 sklearn.feature_extraction.text.CountVectorizer处理文本


import pathlib
import os
import re
import numpy as np


def load_dataset():
    posting_list = ['my dog has flea problems help please',
                    'maybe not take him to dog pack stupid',
                    'my dalmation is so cute I love him',
                    'stop posting stupid worthless garbage',
                    'mr licks ate my steak how to stop him',
                    'quit buying worthless dog food stupid']
    posting_list = [sen.split(' ') for sen in posting_list]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(dataset):
    vocab_set = set([])
    for doc in dataset:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] = 1
        else:
            print('the word : %s is not in my vocabulary!' % word)
    return ret_vec


def train_NB0(train_mat, label):
    num_doc = len(train_mat)
    num_words = len(train_mat[0])
    p_abusive = sum(label)/float(num_doc)  # 1类的先验概率
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)  # 统计两类数据中，各词的词频
    p0_denom, p1_Denom = 2.0, 2.0  # 统计0类和1类的总数
    for i in range(num_doc):
        if label[i] == 1:
            p1_num += train_mat[i]
            p1_Denom += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])
    p1_vect = np.log(p1_num/p1_Denom)
    p0_vect = np.log(p0_num/p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_NB(vec_to_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec_to_classify*p1_vec)+np.log(p_class1)
    p0 = sum(vec_to_classify*p0_vec) + np.log(1.0-p_class1)
    if p1>p0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    list_posts, list_class = load_dataset()

    my_vocabList = create_vocab_list(list_posts)

    train_mat = []

    for post_in_doc in list_posts:

        train_mat.append(set_of_words2vec(my_vocabList, post_in_doc))

    p0V, p1V, pAb = train_NB0(np.array(train_mat), np.array(list_class))

    testEntry = ['love', 'my', 'dalmation']

    thisDoc = np.array(set_of_words2vec(my_vocabList, testEntry))

    print(testEntry, 'classified as: ', classify_NB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']

    thisDoc = np.array(set_of_words2vec(my_vocabList, testEntry))

    print(testEntry, 'classified as: ', classify_NB(thisDoc, p0V, p1V, pAb))
    vocab_list = create_vocab_list(list_posts)
    print(sorted(vocab_list))
    print()
