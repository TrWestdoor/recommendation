# -*- coding: utf-8 -*-

# 基于用户的协同过滤推荐算法实现
# import tensorflow as tf
import random
import sys
import time
import math
from operator import itemgetter


class UserBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = %d' % self.n_rec_movie)


    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename, pivot=0.75):
        trainSet_len = 0
        testSet_len = 0
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                print(line)
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    # 计算用户之间的相似度
    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')

        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')


    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainSet[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]


    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user, in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})
            rec_movies = self.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precision=%.4f \t recall=%.4f \t coverage=%.4f' % (precision, recall, coverage))


if __name__ == '__main__':
    rating_file = './ml-latest-small/ratings.csv'
    userCF = UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()
"""""#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import mpmath
from pandas import DataFrame
import numpy as np
def load():
    #ratings = pd.read_table('/Users/yukino/Downloads/ml-1m/ratings.dat', index_col=None)
    #ratings.describe()
    #ratings.head(5)

    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table('/Users/yukino/Downloads/ml-1m/users.dat', sep='::', header=None, names=unames)

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('/Users/yukino/Downloads/ml-1m/movies.dat', sep='::', header=None, names=mnames)

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('/Users/yukino/Downloads/ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    """
"""unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table('/Users/yukino/Downloads/ml-1m/users.dat', sep='::', header=None, names=unames)
    print(users.shape)
    print(users.head())

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('/Users/yukino/Downloads/ml-1m/movies.dat', sep='::', header=None, names=mnames)
    print(movies.shape)

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('/Users/yukino/Downloads/ml-1m/ratings.dat', sep='::', header=None, names=rnames)
    print(ratings.shape)

    data = pd.merge(pd.merge(ratings, users), movies)
    #data = DataFrame(data=all_data, columns=['user_id', 'movie_id'])

    print(data.shape)
    print(data.columns)
    print(data.head())
    data.to_csv('data.csv')

    trainRatingsDF, testRatingsDF = train_test_split(data, test_size=0.2)

    trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                         index=['userId'], values='rating', fill_value=0)

    moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))

    usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))

    ratingValues = trainRatingsPivotDF.values.tolist()

    userSimMatrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)
    for i in range(len(ratingValues) - 1):
        for j in range(i + 1, len(ratingValues)):
            userSimMatrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
            userSimMatrix[j, i] = userSimMatrix[i, j]
    userMostSimDict = dict()
    for i in range(len(ratingValues)):
        userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[0])), key=lambda x: x[1], reverse=True)[:10]

    userRecommendValues = np.zeros((len(ratingValues), len(ratingValues[0])), dtype=np.float32)
    for i in range(len(ratingValues)):
        for j in range(len(ratingValues[i])):
            if ratingValues[i][j] == 0:
                val = 0
                for (user, sim) in userMostSimDict[I]:
                    val += (ratingValues[user][j] * sim)
                userRecommendValues[i, j] = val

    userRecommendDict = dict()
    for i in range(len(ratingValues)):
        userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:10]

    userRecommendList = []
    for key, value in userRecommendDict.items():
        user = usersMap[key]
        for (movieId, val) in value:
            userRecommendList.append([user, moviesMap[movieId]])

    recommendDF = DataFrame(userRecommendList, columns=['userId', 'movieId'])
    recommendDF = pd.merge(recommendDF, moviesDF[['movieId', 'title']], on='movieId', how='inner')
    recommendDF.tail(10)


    return recommendDF

def calCosineSimilarity(list1 , list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1,val2) in zip(list1,list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (mpmath.sqrt(denominator1 * denominator2))


if __name__ == "__main__":
    data = load(）"""
