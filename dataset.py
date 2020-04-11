import os
import itertools

from collections import defaultdict
# from surprise.trainset import Trainset
from trainset import Trainset


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

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        ur = defaultdict(list)
        ir = defaultdict(list)

        current_u_index = 0
        current_i_index = 0

        for urid, irid, r, timestamp in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = uid
                current_u_index += 1

            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = iid
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)   # number of users
        n_items = len(ir)   # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.reader.rating_scale,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset]


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

