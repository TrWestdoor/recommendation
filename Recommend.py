from dataset import Dataset
from reader import Reader

from surprise.prediction_algorithms import KNNBasic
from model_selection.validation import cross_validate
# from surprise.model_selection import cross_validate


def surprise_code():
    reader = Reader(line_format="user item rating", sep=',', skip_lines=1)
    data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader)
    # print(len(data.raw_ratings))

    algo = KNNBasic()
    perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=2, verbose=0)
    print(perf)


if __name__ == '__main__':
    surprise_code()
