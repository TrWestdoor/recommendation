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

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}





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

