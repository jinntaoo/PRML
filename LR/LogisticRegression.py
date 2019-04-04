# @author: JinnTaoo
# @email: jinntaoor@gmail.com


from sklearn import datasets
import numpy as np
import random


def load_data():
    """
    使用iris数据集，使用前两类
    :param
    :return: 训练和测试数据的样本及其标签
    """
    iris = datasets.load_iris()
    samples = iris.data.tolist()
    labels = iris.target.tolist()
    samples_ = list()
    for sample, label in zip(samples, labels):
        if label == 0 or label == 1:
            samples_.append([sample, label])
    random.shuffle(samples_)
    train_ = samples_[:int(len(samples_)*0.8)]
    test_ = samples_[int(len(samples_)*0.8):]

    train_samples, train_labels = np.array(list(np.array(train_)[:, 0])), np.reshape(np.array(list(np.array(train_)[:, 1])), (80, 1))
    test_samples, test_labels = np.array(list(np.array(test_)[:, 0])), np.reshape(np.array(list(np.array(test_)[:, 1])), (20, 1))

    return train_samples, train_labels, test_samples, test_labels


def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def grad_descent(samples, labels, steps, alpha):
    """
    梯度下降法,且vectorization：
    (a)A=x*theta; (b)error=h(a)-y; (c)theta=theta-alpha*x.T*error
    :param samples: 数据
    :param labels: 标签
    :param steps: 最大学习步数
    :param alpha: 学习率
    :return: 模型w
    """
    m, n = samples.shape
    weights = np.zeros((n, 1))
    for step in range(steps):
        A = np.dot(samples, weights)
        error = sigmoid(A) - labels
        weights -= alpha * np.dot(samples.T, error)
    return weights


def stochastic_gradient_decent(samples, labels, epochs, alpha):
    # samples, labels = samples.tolsit(), labels.tolist()
    m, n = samples.shape
    weights = np.ones((n, 1))
    rand_index = range(len(labels))

    for epoch in range(epochs):
        selected_index = list(range(len(labels)))
        random.shuffle(selected_index)
        for step in range(m):
            alpha = 0.001/(1.0+epoch)+0.0001
            A = np.dot(samples[rand_index[step]].reshape((1, -1)), weights)
            error = sigmoid(A) - labels[selected_index[step]]
            weights -= alpha * np.dot(samples[rand_index[step]].reshape((-1, 1)), error)
    return weights


def predict(weights, a_sample):
    z = np.dot(a_sample, weights)
    p1 = np.exp(z)/(1+np.exp(z))
    if p1 > 0.5:
        return 1
    return 0


def test(samples, labels, weights):
    n, m = samples.shape
    num_error = 0
    for sample, label in zip(samples, labels):
        pred = predict(weights, sample)
        if label != pred:
            num_error += 1
    return 1-(num_error/n)


if __name__ == "__main__":
    import time
    print('loading samples')
    train_samples, train_labels, test_samples, test_labels = load_data()
    print('traing---')
    start = time.time()

    weights = stochastic_gradient_decent(train_samples, train_labels, epochs=50, alpha=0.001)
    # weights = grad_descent(train_samples, train_labels, steps=5, alpha=0.001)

    print('done training, use time: ', int(time.time()-start))
    acc = test(test_samples, test_labels, weights)
    print('test acc: ', acc)
