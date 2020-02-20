import random
import re
import os
import itertools


class Dataset:
    """
    Base class for loading datasets.
    """
    def __init__(self, reader):
        self.reader = reader

    @classmethod
    def load_from_file(cls, file_path, reader):
        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    def read_ratings(self, filename):
        """
        Return a list of ratings (user, item, rating, timestamp) read from file_name
        """
        with open(os.path.expanduser(filename)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]

        return raw_ratings


class DatasetAutoFolds(Dataset):
    """

    """
    def __init__(self, ratings_file=None, reader=None, df=None):
        Dataset.__init__(self, reader)

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)

        elif df is not None:
            pass

        else:
            raise ValueError('Must specify ratings file or dataframe.')

    @classmethod
    def data_load(cls, file_path, seed, m):
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
                if random.randint(1, m) == 1:
                    if user not in test:
                        test[user] = []
                    test[user].append(movie)
                else:
                    if user not in train:
                        train[user] = []
                    train[user].append(movie)
        return test, train

