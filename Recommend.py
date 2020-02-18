import math
import re
import random
from operator import itemgetter


class Dataset:
    """
    Base class for loading datasets.
    """
    def data_load(self, file_path, seed, m):
        """
        input: file_path: data set path
        output: {{item: rating}, {item: rating},...{item: rating}}
        """
        test = dict()
        train = dict()
        # 数据集以dict的方式存储，key为user id， value为user看过的movie id组成的list
        random.seed(seed)
        with open(file_path, 'r') as f:
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
                if random.randint(1, M) == 1:
                    if user not in test:
                        test[user] = []
                    test[user].append(movie)
                else:
                    if user not in train:
                        train[user] = []
                    train[user].append(movie)
        return test, train


if __name__ == '__main__':
    data = data_load('')
    algo = UserCF()
    result = train(data=data, Algo=algo)
    test_result = evaluate(result, test_data)
    pass
