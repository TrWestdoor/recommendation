# -*- coding: utf-8 -*-
import math
import random
import re
from operator import itemgetter

REC_NUMBER = 10
similarity_user = 50


# 读取数据，分割为训练集和测试集：M表示 1/M 的样本为测试集
def split_data(filename, M, seed):
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
            user = temp[0]
            movie = temp[1]
            rating = temp[2]
            user = int(user)

            # 此处int(rating)会报错；4分以下的评分忽略
            if float(rating) < 4:
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


def UserSimilarity(train):
    # build inverse table for item_users: it's shown that each item was selected by which users
    item_users = {}
    for u, items in train.items():
        for i in items:
            if i not in item_users:
                item_users[i] = []
            item_users[i].append(u)

    user_sim_matrix = {}
    for i, users in item_users.items():
        for u in users:
            for v in users:
                if u == v:
                    continue
                user_sim_matrix.setdefault(u, {})
                user_sim_matrix[u].setdefault(v, 0)
                user_sim_matrix[u][v] += 1

    # calculate finial similarity matrix W
    for u, related_users in user_sim_matrix.items():
        for v, cuv in related_users.items():
            user_sim_matrix[u][v] = cuv / math.sqrt(len(train[u]) * len(train[v]))
    return user_sim_matrix


def Recommend(user, train, W):
    rank = dict()
    k = similarity_user
    watched_items = train[user]
    # obtained most k user which similarity given user
    for v, wuv in sorted(W[user].items(),key=itemgetter(1), reverse=True)[:k]:
        for movie in train[v]:
            if movie in watched_items:
                continue
            rank.setdefault(movie, 0)
            rank[movie] += wuv
    # return most n movies which have large probability given user will like these.
    # Return type: [{movieID: similarity score}, {},...,{}]
    return sorted(rank.items(), key=itemgetter(1), reverse=True)


class Evaluation:

    def __init__(self, train, test, W):
        self.train = train
        self.test = test
        self.W = W
        self.N = REC_NUMBER
        self.hit = 0
        self.hit_filter = 0
        self.all = 0
        self.recommend_items = set()
        self.recommend_items_filter = set()
        self.all_items = set()
        readfile = '.'
        # with open(readfile + '/Pmatrix.json', 'r') as f:
        #     self.pmatrix = json.loads(f.read())
        # with open(readfile + '/Qmatrix.json', 'r') as f:
        #     self.qmatrix = json.loads(f.read())
        self.pmatrix = self.load_mf_matrix(readfile + '/11_1_p.csv')
        self.qmatrix = self.load_mf_matrix(readfile + '/11_1_q.csv')

    @staticmethod
    def load_mf_matrix(target_file):
        with open(target_file) as f:
            temp_matrix = {}
            f.readline()
            for line in f.readlines():
                line = line.strip().split(',')
                attr = {}
                for i in range(5):
                    attr[str(i)] = line[i+1]
                temp_matrix[line[0]] = attr
        return temp_matrix

    def run(self):
        for user in self.train.keys():
            for item in self.train[user]:
                self.all_items.add(item)
            tu = self.test.get(user, {})  # user corresponding movie list in the test.
            Init_rank = Recommend(user, self.train, self.W)
            rank = Init_rank[:self.N]
            filter_rank = self.Filter(user, Init_rank)[:self.N]
            for item, _ in rank:
                if item in tu:
                    self.hit += 1
                self.recommend_items.add(item)
            self.all += len(tu)

            for item, _ in filter_rank:
                if item in tu:
                    self.hit_filter += 1
                self.recommend_items_filter.add(item)

    def Filter(self, user, InitRank):
        user = str(user)
        user_att = sorted(self.pmatrix[user].items(), key=itemgetter(1))
        NegativeAtt = []
        for i in range(5):
            negative, _ = user_att[i]
            NegativeAtt.append(negative)
        WillRm = []
        for item in self.qmatrix.keys():
            item_att = sorted(self.qmatrix[item].items(), key=itemgetter(1), reverse=True)[:5]
            for att, _ in item_att:
                if att in NegativeAtt:
                    WillRm.append(item)
                    break
        for item, scoring in InitRank:
            for will_rm in WillRm:
                if item == will_rm:
                    InitRank.remove((item, scoring))
        return InitRank

    def Recall(self):
        print('Recall: ', self.hit / (self.all * 1.0))
        print('After filtering: ', self.hit_filter / (self.all * 1.0))

    def Precision(self):
        print('Precision: ', self.hit / (len(self.train) * self.N * 1.0))
        print('After filtering: ', self.hit_filter / (len(self.train) * self.N * 1.0))

    def Coverage(self):
        print('Coverage: ', len(self.recommend_items) / (len(self.all_items) * 1.0))
        print('After filtering: ', len(self.recommend_items_filter) / (len(self.all_items) * 1.0))


def main():
    filename = './ml-latest-small/ratings.csv'
    test, train = split_data(filename, 5, random.random())
    W = UserSimilarity(train)
    result = Evaluation(train, test, W)
    result.run()
    result.Precision()
    result.Recall()
    result.Coverage()


if __name__ == '__main__':
    main()
