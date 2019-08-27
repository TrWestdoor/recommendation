# coding:utf-8
import math
import random
import re
import operator


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


def pre_data(filename):

    test, user_items= SplitData(filename, 8, 10)
    users = set(user_items.keys())
    items = [item for user_item in user_items.values() for item in user_item]
    items = set(items)
    print(users)
    print(items)
    return user_items, users, items, test


def initpara(users, items, F):
    p = dict()
    q = dict()

    for userid in users:
        p[userid] = [(-1 + 2 * random.random()) for f in range(0, F)]  # / math.sqrt(F)

    for itemid in items:
        q[itemid] = [(-1 + 2 * random.random()) for f in range(0, F)]  # / math.sqrt(F)

    return p, q


def initsamples(user_items):
    user_samples = []
    items_pool = []
    for userid, items in user_items.items():
        for item in items:
            items_pool.append(item)

    for userid, items in user_items.items():
        samples = dict()
        for itemid, score in items.items():
            if score != 0:
                samples[itemid] = score
        user_samples.append((userid, samples))

    return user_samples


def initmodel(user_items, users, items, F):
    p, q = initpara(users, items, F)
    user_samples = initsamples(user_items)

    return p, q, user_samples


def predict(userid, itemid, p, q):
    a = sum(p[userid][f] * q[itemid][f] for f in range(0, len(p[userid])))
    return a


def lfm(user_items, users, items, F, N, alpha, lamda):
    '''
    LFM计算参数 p,q
    :param user_items: user_items
    :param users: users
    :param items: items
    :param F: 隐类因子个数
    :param N: 迭代次数
    :param alpha: 步长
    :param lamda: 正则化参数
    :return: p,q
    '''
    p, q, user_samples = initmodel(user_items, users, items, F)

    debugid1 = 0
    debugid2 = 0
    for step in range(0, N):
        random.shuffle(user_samples)  # 乱序

        error = 0
        count = 0
        for userid, samples in user_samples:
            for itemid, rui in samples.items():
                pui = predict(userid, itemid, p, q)
                eui = rui - pui
                count += 1
                # error += math.pow(eui, 2)
                error += abs(eui)
                '''debug'''
                if userid == 1:
                    if debugid1 == 0 and rui == 1:
                        debugid1 = itemid
                    if debugid2 == 0 and rui == -1:
                        debugid2 = itemid

                if userid == 1 and itemid == debugid1:
                    print(debugid1, rui, pui, eui, alpha)
                if userid == 1 and itemid == debugid2:
                    print(debugid2, rui, pui, eui, alpha)

                '''debug end'''

                for f in range(0, F):
                    p_u = p[userid][f]
                    q_i = q[itemid][f]

                    p[userid][f] += alpha * (eui * q_i - lamda * p_u)
                    q[itemid][f] += alpha * (eui * p_u - lamda * q_i)

        # rmse = math.sqrt(error / count)
        # print("rmse:", rmse)
        alpha *= 0.9
        print("error: ", error)
    return p, q


def predictlist(userid, items, p, q):
    predict_score = dict()
    for itemid in items:
        p_score = predict(userid, itemid, p, q)
        predict_score[itemid] = p_score

    return predict_score


def train_result(user_items, p, q):
    # user_items = {1: {'a': 1, 'b': -1, 'c': -1, 'd': -1, 'e': 1, 'f': 1, 'g': -1},
    #               2: {'a': -1, 'b': 1, 'c': -1, 'd': 1, 'e': 1, 'f': 1, 'g': 1},
    #               3: {'a': 1, 'b': -1, 'c': 0, 'd': -1, 'e': -1, 'f': -1, 'g': 1},
    #               4: {'a': 1, 'b': -1, 'c': -1, 'd': 0, 'e': 1, 'f': 1, 'g': 1},
    #               5: {'a': -1, 'b': 1, 'c': 1, 'd': 1, 'e': -1, 'f': -1, 'g': 0},
    #               6: {'a': 1, 'b': 0, 'c': -1, 'd': -1, 'e': 1, 'f': -1, 'g': -1}}
    # users = {1, 2, 3, 4, 5, 6}
    # items = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
    # for values in user_items.values():
    #     for item in items:
    #         if values[item] != 1:
    #             values.pop(item)
    # print(user_items)
    #
    # F = 5
    # N = 30
    # alpha = 0.3
    # lamda = 0.03
    # p, q = lfm(user_items, users, items, F, N, alpha, lamda)

    for userid, itemdic in user_items.items():
        print(userid)
        print("target", itemdic)
        predict_score = predictlist(userid, itemdic, p, q)
        print("predicted", predict_score)

    return


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

    def Recommend(P, Q, user, train):
        n = REC_NUMBER
        rank = dict()
        interacted_items = train[user]
        for i in range(np.shape(Q)[0]):
            if i in interacted_items.keys():
                continue
            ri = 0
            # for i in range()
            # ri = np.dot(P[user - 1], Q[i])
            rank.setdefault(i, ri)

        return sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:n]

    def run(self):
        for user in self.train.keys():
            tu = self.test.get(user, {})  # user corresponding movie list in the test.
            rank = self.Recommend(self.p, self.q, user, self.train)

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


def evaluation(train, test, p, q):
    result = Evaluation(train, test, p, q)
    result.run()
    print('precision: ', result.Precision())
    print('recall: ', result.Recall())
    print('coverage ', result.Coverage())


def main():
    print('start')
    filename = './ml-latest-small/ratings.csv'
    user_items, users, items, test = pre_data(filename)

    F = 10
    N = 30
    alpha = 0.3
    lamda = 0.03
    p, q = lfm(user_items, users, items, F, N, alpha, lamda)

    train_result(user_items, p, q)
    print(p)
    # evaluation()

    print('end')
    return


if __name__ == "__main__":
    REC_NUMBER = 10
    # train_result()
    main()
