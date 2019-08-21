# -*- coding: utf-8 -*-
"""
Created on Fri Nov  18 21:54:00 2018

@author: TrWestdoor
"""

import random
import operator
import json
import re
import numpy as np


REC_NUMBER = 10

allItemSet = set()


def SplitData(filename, M, seed):
    # 读取数据，分割为训练集和测试集：M表示 1/M 的样本为测试集
    test = dict()
    train = dict()
    # 数据集以dict的方式存储，key为user id， value为user看过的movie id组成的list
    random.seed(seed)
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            # user, movie, rating, timestamp = line.split(' ')
            temp = re.split(',', line)
            user = int(temp[0])
            movie = int(temp[1])
            rating = temp[2]
            # user = int(user)

            # 此处int(rating)会报错；4分以下的评分忽略
            if float(rating) < 4:
                continue

            # 分割数据集
            if random.randint(1,M) == 1:
                test.setdefault(user, {})
                test[user][movie] = 1
            else:
                train.setdefault(user, {})
                train[user][movie] = 1
    return test, train


def InitAllItemSet(user_items):
    allItemSet.clear()
    for user, items in user_items.items():
        for i, r in items.items():
            allItemSet.add(i)


def InitItems_Pool(items):
    interacted_items = set(items.keys())
    items_pool = list(allItemSet - interacted_items)
    #    items_pool = list(allItemSet)
    return items_pool
    # input some watched movies with dict format,
    # and return remain movies in all movies set with list format


def RandSelectNegativeSample(items):
    ret = dict()
    for i in items.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(items) * 3):
        items_pool = InitItems_Pool(items)
        item = items_pool[random.randint(0, len(items_pool) - 1)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret


def Predict(user, item, P, Q):
    rate = 0
    for f, puf in P[user].items():
        qif = Q[item][f]
        rate += puf * qif
    return rate


# def InitModel(user_items, F):
#     P = dict()
#     Q = dict()
#     for user, items in user_items.items():
#         P[user] = dict()
#         for f in range(0, F):
#             P[user][f] = random.random()
#         for i, r in items.items():
#             if i not in Q:
#                 Q[i] = dict()
#                 for f in range(0, F):
#                     Q[i][f] = random.random()
#     print(len(P), len(Q))
#     return P, Q


def init_model(user_items, attribute_num):
    user_num = max(user_items.keys())
    item_list = []
    for items in user_items.values():
        for item in items.keys():
            # print(item)
            if item not in item_list:
                item_list.append(item)
    p_mat = np.random.random((user_num, attribute_num))
    q_mat = np.random.random((max(item_list), attribute_num))
    return p_mat, q_mat


# dict{userID: {}}, F:latent factor, echo nubmer, learning rate, lambda
def LatentFactorModel(user_items, F, T, alpha, lamb):
    InitAllItemSet(user_items)
    # P, Q are dict, and every value also are dict
    # [P, Q] = InitModel(user_items, F)
    P, Q = init_model(user_items, F)

    loop = 0
    # training loop
    for step in range(0, T):
        total_err = 0
        for user, items in user_items.items():
            # not only negative sample, also have positive samples
            samples = RandSelectNegativeSample(items)
            for item, rui in samples.items():
                # eui = rui - Predict(user, item, P, Q)
                eui = rui - np.dot(P[user-1], Q[item-1])
                total_err += abs(eui)
                for f in range(0, F):
                    P[user-1][f] += alpha * (eui * Q[item-1][f] - lamb * P[user-1][f])
                    Q[item-1][f] += alpha * (eui * P[user-1][f] - lamb * Q[item-1][f])
        alpha *= 0.9
        loop += 1
        print(loop, ':', total_err)
    return P, Q


def Recommend(P, Q, user, train):
    n = REC_NUMBER
    rank = dict()
    interacted_items = train[user]
    for i in range(np.shape(Q)[0]):
        if i in interacted_items.keys():
            continue
        ri = np.dot(P[user-1], Q[i])
        rank.setdefault(i, ri)

    return sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:n]


def Recommendation(users, train, P, Q):
    result = dict()
    for user in users:
        rank = Recommend(user, train, P, Q)
        R = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
        result[user] = R
    return result


class Evaluation:

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
    test, train = SplitData(filename, 8, 10)
    # LatentFactorModel function input: train, F, epoch, alpha, lambda
    p, q = LatentFactorModel(train, 50, 2, 0.02, 0.01)
    result = Evaluation(train, test, p, q)
    result.run()
    print('precision: ', result.Precision())
    print('recall: ', result.Recall())
    print('coverage ', result.Coverage())
    with open('./Pmatrix.json', 'w') as f:
        pjson = json.dumps(p)
        f.write(pjson)
        p = json.loads(pjson)
    with open('./Qmatrix.json', 'w') as f:
        qjson = json.dumps(q)
        f.write(qjson)


if __name__ == '__main__':
    main()
