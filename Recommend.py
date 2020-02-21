import math
from operator import itemgetter
from dataset import Dataset
from reader import Reader
from surprise.prediction_algorithms import KNNBasic
from surprise.model_selection import cross_validate


if __name__ == '__main__':
    reader = Reader(line_format="user item rating", sep=',', skip_lines=1)
    data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader)
    # print(len(data.raw_ratings))

    measure = 'mse'
    algo = KNNBasic()
    perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
    print(perf)
