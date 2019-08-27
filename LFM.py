# -*- coding: utf-8 -*-
import math
import numpy as np
import random
import re
from operator import itemgetter

REC_NUMBER = 10
F = 10  # latent factor numbers
ALPHA = 0.02


def SplitData(filename, M, seed):           # 读取数据，分割为训练集和测试集：M表示 1/M 的样本为测试集
    test = dict()
    train = dict()                          # 数据集以dict的方式存储，key为user id， value为user看过的movie id组成的list
    random.seed(seed)
    with open(filename,'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            # user, movie, rating, timestamp = line.split(' ')
            temp = re.split(',', line)
            user = temp[0]
            movie = temp[1]
            rating = temp[2]
            user = int(user)
            if float(rating) < 4:                   # 此处int(rating)会报错；4分以下的评分忽略
                continue
            if random.randint(1,M) == 1:
                if user not in test:
                    test[user] = []
                test[user].append(movie)
            else:
                if user not in train:
                    train[user] = []
                train[user].append(movie)
    return test, train


def InitLFM(train):
    p = dict()
    q = dict()
    for u in train.keys():
        if u not in p:
            p[u] = np.random.rand(F)
        for i in train[u]:
            if i not in q:
                q[i] = np.random.rand(F)
    return p, q


def LearningLFM(train, n, lam):
    (p,q) = InitLFM(train)
    alpha = ALPHA
    for step in range(0, n):
        total_error = 0.0
        for u in train:
            for i in q:
                if i in train[u]:
                    rui = 1
                    pui = np.dot(p[u], q[i])
                    eui = rui - pui
                    total_error += np.abs(eui)
                    p[u] += alpha * (q[i] * eui - lam * p[u])
                    q[i] += alpha * (p[u] * eui - lam * q[i])
        alpha *= 0.9
        print(step, ':',total_error, 'alpha:', alpha)

    return p, q


def Recommend(p, q, user, train):
    n = REC_NUMBER
    rank = {}
    for i in q:
        if i in train[user]:
            continue
        rank.setdefault(i, 0)
        rank[i] = np.dot(p[user], q[i])
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[:n]


class Evaluation():

    def __init__(self, train, test, p, q):
        self.train = train
        self.test = test
        self.p = p
        self.q = q
        self.N = REC_NUMBER
        self.hit = 0
        self.all = 0
        self.recommend_items = set()

    def run(self):
        for user in self.train.keys():
            tu = self.test.get(user, {})  # user corresponding movie list in the test.
            rank = Recommend(self.p, self.q, user, self.train)

            for item, _ in rank:
                if item in tu:
                    self.hit += 1
                self.recommend_items.add(item)
            self.all += len(tu)

    def Recall(self):
        return self.hit / (self.all * 1.0)

    def Precision(self):
        return self.hit / (len(self.train) * self.N * 1.0)

    def Coverage(self):
        return len(self.recommend_items) / (len(self.q) * 1.0)


def main():
    filename = './ml-latest-small/ratings.csv'
    test, train = SplitData(filename, 5, 10)

    # input: train, epoch, lambda
    (p, q) = LearningLFM(train, 100, 0.01)
    result = Evaluation(train, test, p, q)
    result.run()
    print('precision: ', result.Precision())
    print('recall: ', result.Recall())
    print('coverage ', result.Coverage())


if __name__ == '__main__':
    main()




