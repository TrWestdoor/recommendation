# -*- coding: utf-8 -*-
"""
Created on Fri Nov  18 21:54:00 2018

@author: TrWestdoor
"""

import random
import operator
import json
import re

REC_NUMBER = 10

allItemSet = set()


def SplitData(filename, M, seed):           #读取数据，分割为训练集和测试集：M表示 1/M 的样本为测试集
    test = dict()
    train = dict()                          #数据集以dict的方式存储，key为user id， value为user看过的movie id组成的list
    random.seed(seed)
    with open(filename,'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            #user, movie, rating, timestamp = line.split(' ')
            temp = re.split('\s+', line)
            user = temp[0]; movie = temp[1]; rating = temp[2]
            user = int(user)
            if float(rating) < 4:                   #此处int(rating)会报错；4分以下的评分忽略
                continue
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
    return items_pool           #input some watched movies with dict format, and return reamain movies in all movies set with list format


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


def InitModel(user_items, F):
    P = dict()
    Q = dict()
    for user, items in user_items.items():
        P[user] = dict()
        for f in range(0, F):
            P[user][f] = random.random()
        for i, r in items.items():
            if i not in Q:
                Q[i] = dict()
                for f in range(0, F):
                    Q[i][f] = random.random()
    return P, Q

def LatentFactorModel(user_items, F, T, alpha, lamb):   #dict{userID: {}}, F:latent factor, echo nubmer, learning rate, lambda
    InitAllItemSet(user_items)
    # P, Q are dict, and every value also are dict
    [P, Q] = InitModel(user_items, F)
    looop = 0
    for step in range(0, T):                    #training loop
        for user, items in user_items.items():
            # not only negative sample, also have positive samples
            samples = RandSelectNegativeSample(items)
            for item, rui in samples.items():
                eui = rui - Predict(user, item, P, Q)
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - lamb * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - lamb * Q[item][f])
        alpha *= 0.9
        looop += 1
        print(looop)
    return P, Q


def Recommend(P, Q, user, train):
    n = REC_NUMBER
    rank = dict()
    interacted_items = train[user]
    for i in Q:
        if i in interacted_items.keys():
            continue
        rank.setdefault(i, 0)
        for f, qif in Q[i].items():
            puf = P[user][f]
            rank[i] += puf * qif
    return sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:n]


def Recommendation(users, train, P, Q):
    result = dict()
    for user in users:
        rank = Recommend(user, train, P, Q)
        R = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)
        result[user] = R
    return result

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

if __name__ == '__main__':
    filename = './ml-100k/u.data'
    test, train = SplitData(filename, 5, 10)
    (p, q) = LatentFactorModel(train, 100, 3000, 0.02, 0.01)        #input: train, F, epcho, alpha, lambda
    result = Evaluation(train, test, p, q)
    result.run()
    print('precision: ', result.Precision())
    print('recall: ', result.Recall())
    print('coverage ', result.Coverage())
    with open('./Pmatrix.json', 'w') as f:
        pjson = json.dumps(p)
        f.write(pjson)
        p = json.loads(pjson)
        print(len(p))
        print(type(p))
    with open('./Qmatrix.json', 'w') as f:
        qjson = json.dumps(q)
        f.write(qjson)