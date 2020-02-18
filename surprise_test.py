import os
from surprise import SVD, KNNBaseline
from surprise import Dataset, Reader
# from surprise import print_perf
from surprise.model_selection import cross_validate

# data = Dataset.load_builtin('ml-100k')
# algo = SVD()
#
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


file_path = os.path.expanduser('./ml-latest-small/ratings.csv')
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
# data = Dataset(reader)
# data = data.load_from_file(file_path, reader=reader)
data = Dataset.load_from_file(file_path, reader=reader)
algo = KNNBaseline()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)

print(perf)
