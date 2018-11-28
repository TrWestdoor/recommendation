# -*- coding: utf-8 -*-
"""
Created on Fri Nov  18 21:54:00 2018

@author: TrWestdoor
"""

import random
import operator
import json
import re
import pdb

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
#            user, movie, rating, timestamp = line.split(' ')
            temp = re.split('\s+', line)
            user = temp[0]; movie = temp[1]; rating = temp[2]
            user = int(user)
            if random.randint(1,M) == 1:
                test.setdefault(user, {})
                test[user][movie] = rating
            else:
                train.setdefault(user, {})
                train[user][movie] = rating
    return test, train


def InitAllItemSet(user_items):
    allItemSet.clear()
    for user, items in user_items.items():
        for i, r in items.items():
            allItemSet.add(i)



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
        totalerr = 0
        for user, items in user_items.items():
            for item, rui in items.items():
                eui = int(rui) - Predict(user, item, P, Q)
                totalerr += abs(eui)
                print(eui)
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - lamb * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - lamb * Q[item][f])
        alpha *= 0.99
        looop += 1
        print(looop, ':', totalerr, type(totalerr))
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
    test, train = SplitData(filename, 8, random.random())
    print(test)
#    pdb.set_trace()
    (p, q) = LatentFactorModel(train, 20, 100, 0.05, 0.01)        #input: train, F, epcho, alpha, lambda
    result = Evaluation(train, test, p, q)
    result.run()
    print('precision: ', result.Precision())
    print('recall: ', result.Recall())
    print('coverage ', result.Coverage())
    writefile = './Result/11.19/2'
    with open(writefile + '/Pmatrix.json', 'w') as f:
        pjson = json.dumps(p)
        f.write(pjson)
        p = json.loads(pjson)
    with open(writefile + '/Qmatrix.json', 'w') as f:
        qjson = json.dumps(q)
        f.write(qjson)