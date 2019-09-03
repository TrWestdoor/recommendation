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

    for user in test.keys():
        users.add(user)
    for test_item in test.values():
        for item in test_item:
            items.add(item)
    # print(users)
    # print(items)
    return user_items, users, items, test


def initpara(users, items, F):
    p = dict()
    q = dict()

    for userid in users:
        p[userid] = [(-1 + 2 * random.random()) for f in range(0, F)]  # / math.sqrt(F)

    for itemid in items:
        q[itemid] = [(-1 + 2 * random.random()) for f in range(0, F)]  # / math.sqrt(F)

    return p, q


def rand_select_negative_sample(items_pool, positive_samples):
    ret = dict()
    for i in positive_samples.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(positive_samples) * 3):
        item = items_pool[random.randint(0, len(items_pool) - 1)]
        if item in ret or item not in items_pool:
            continue
        ret[item] = 0
        n += 1
        if n > len(positive_samples):
            break
    return ret


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

        print("init samples: ", samples)
        # add negative samples
        new_samples = rand_select_negative_sample(items_pool, samples)
        print("add negative samples: ", new_samples)

        user_samples.append((userid, new_samples))

    return user_samples


def initmodel(user_items, users, items, F):
    p, q = initpara(users, items, F)
    user_samples = initsamples(user_items)

    return p, q, user_samples


def predict(userid, itemid, p, q):
    a = sum(p[userid][f] * q[itemid][f] for f in range(0, len(p[userid])))
    return a


def lfm(user_items, users, items, F, N, alpha, lamda):
    """
    LFM计算参数 p,q
    :param user_items: user_items
    :param users: users
    :param items: items
    :param F: 隐类因子个数
    :param N: 迭代次数
    :param alpha: 步长
    :param lamda: 正则化参数
    :return: p,q
    """
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
                # if userid == 1:
                #     if debugid1 == 0 and rui == 1:
                #         debugid1 = itemid
                #     if debugid2 == 0 and rui == -1:
                #         debugid2 = itemid
                #
                # if userid == 1 and itemid == debugid1:
                #     print(debugid1, rui, pui, eui, alpha)
                # if userid == 1 and itemid == debugid2:
                #     print(debugid2, rui, pui, eui, alpha)

                '''debug end'''

                for f in range(0, F):
                    p_u = p[userid][f]
                    q_i = q[itemid][f]

                    p[userid][f] += alpha * (eui * q_i - lamda * p_u)
                    q[itemid][f] += alpha * (eui * p_u - lamda * q_i)

        # rmse = math.sqrt(error / count)
        # print("rmse:", rmse)
        # alpha *= 0.95
        print("error: ", error)
    return p, q


def predictlist(userid, items, p, q):
    predict_score = dict()
    for itemid in items:
        p_score = predict(userid, itemid, p, q)
        predict_score[itemid] = p_score

    return predict_score


def train_result(user_items, items, p, q):
    for userid, itemdic in user_items.items():
        print("user id: ", userid)
        print("target", itemdic)
        predict_score = predictlist(userid, itemdic, p, q)
        print("predicted", predict_score)

        # check unknown item score
        # unknown_item = [random.randint(0, 10000) for i in range(10)]
        items_list = list(items)
        unknown_item = [items_list[random.randint(0, len(items)-1)] for _ in range(10)]
        for i in unknown_item:
            if i not in items:
                unknown_item.remove(i)
        print("unknown item score: ", predictlist(userid, unknown_item, p, q))
    return


class Evaluation:

    def __init__(self, train, test, p, q, items):
        self.train = train
        self.test = test
        self.p = p
        self.q = q
        self.N = REC_NUMBER
        self.hit = 0
        self.all = 0
        self.recommend_items = set()
        self.items = items

    def Recommend(self, P, Q, user, train):
        n = REC_NUMBER
        rank = dict()
        interacted_items = train[user]
        for i in self.items:
            if i in interacted_items.keys():
                continue
            # ri = np.dot(P[user - 1], Q[i])
            ri = predict(user, i, P, Q)
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


def evaluation(train, test, p, q, items):
    result = Evaluation(train, test, p, q, items)
    result.run()
    print('precision: ', result.Precision())
    print('recall: ', result.Recall())
    print('coverage ', result.Coverage())


def main():
    print('start')
    filename = './ml-latest-small/ratings.csv'
    user_items, users, items, test = pre_data(filename)

    F = 30
    N = 100
    alpha = 0.1
    lamda = 0.05
    p, q = lfm(user_items, users, items, F, N, alpha, lamda)

    # train_result(user_items, items, p, q)
    # print(p)
    evaluation(user_items, test, p, q, items)

    print('end')
    return


if __name__ == "__main__":
    REC_NUMBER = 10
    # train_result()
    main()
