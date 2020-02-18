class Reader:
    """The Reader class is used to parse a file containing ratings."""
    def __init__(self, name=None, line_format='user item rating', sep=None,
                 rating_scale=(1, 5), skip_lines=0):

        if name:
            pass

        else:
            self.sep = sep
            self.rating_scale = rating_scale
            self.skip_lines = skip_lines

            split_format = line_format.split()
            entities = ['user', 'item', 'rating']
            if 'timestamp' in split_format:
                entities.append('timestamp')
                self.with_timestamp = True
            else:
                self.with_timestamp = False

            if any(filed not in entities for filed in split_format):
                raise ValueError('line_format parameter is incorrect.')

            self.indexes = [split_format.index(entity) for entity in
                            entities]

    def parse_line(self, line):
        """Parse a line.

        Ratings are translated so that they are all strictly positive.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, rating and timestamp. The timestamp is set
            to ``None`` if it does no exist.
        """

        line = line.split(self.sep)
        try:
            if self.with_timestamp:
                uid, iid, r, timestamp = (line[i].strip()
                                          for i in self.indexes)
            else:
                uid, iid, r = (line[i].strip()
                               for i in self.indexes)
                timestamp = None

        except IndexError:
            raise ValueError('Impossible to parse line. Check the line_format'
                             ' and sep parameters.')

        return uid, iid, float(r), timestamp
