"""This module contains the Trainset class."""

import numpy as np
from six import iteritems


class Trainset:
    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 raw2inner_id_users, raw2inner_id_items):
        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None

        self._inner2raw_id_users = None
        self._inner2raw_id_items = None

    def knows_user(self, uid):
        return uid in self.ur

    def knows_item(self, iid):
        return iid in self.ir

    def to_inner_uid(self, ruid):
        try:
            return self._raw2inner_id_users[ruid]

        except KeyError:
            raise ValueError("User " + str(ruid) + " is not part of the trainset")

    def to_raw_uid(self, iuid):
        """Convert a **user** inner id to a raw id.
                Args:
                    iuid(int): The user inner id.

                Returns:
                    str: The user raw id.

                Raises:
                    ValueError: When ``iuid`` is not an inner id.
                """

        """If inner2raw dict is None, construct an inner2raw id dict from raw2inner dict."""
        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_users)}

        try:
            return self._inner2raw_id_items[iuid]

        except KeyError:
            raise ValueError(str(iuid) + " is not a valid inner id")

    def to_inner_iid(self, riid):
        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError("Item " + str(riid) + " is not part of the trainset.")

    def to_raw_iid(self, iiid):
        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_items)}

        try:
            return self._inner2raw_id_users[iiid]
        except KeyError:
            raise ValueError(str(iiid) + " is not a valid id")

    def all_ratings(self):
        """Generator function to iterate over all ratings.

                Yields:
                    A tuple ``(uid, iid, rating)`` where ids are inner ids (see
                    :ref:`this note <raw_inner_note>`).
                """
        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

    def build_testset(self):
        """Return a list of ratings that can be used as a testset in the
        :meth:`test() <surprise.prediction_algorithms.algo_base.AlgoBase.test>`
        method.

        The ratings are all the ratings that are in the trainset, i.e. all the
        ratings returned by the :meth:`all_ratings()
        <surprise.Trainset.all_ratings>` generator. This is useful in
        cases where you want to to test your algorithm on the trainset.
        """
        return [(self.to_raw_uid(u), self.to_raw_iid(i), r)
                for (u, i, r) in self.all_ratings()]

    def build_anti_testset(self, fill=None):
        """Return a list of ratings that can be used as a testset in the
        :meth:`test() <surprise.prediction_algorithms.algo_base.AlgoBase.test>`
        method.

        The ratings are all the ratings that are **not** in the trainset, i.e.
        all the ratings :math:`r_{ui}` where the user :math:`u` is known, the
        item :math:`i` is known, but the rating :math:`r_{ui}`  is not in the
        trainset. As :math:`r_{ui}` is unknown, it is either replaced by the
        :code:`fill` value or assumed to be equal to the mean of all ratings
        :meth:`global_mean <surprise.Trainset.global_mean>`.

        Args:
            fill(float): The value to fill unknown ratings. If :code:`None` the
                global mean of all ratings :meth:`global_mean
                <surprise.Trainset.global_mean>` will be used.

        Returns:
            A list of tuples ``(uid, iid, fill)`` where ids are raw ids.
        """

        fill = self.global_mean if fill is None else fill

        anti_testset = []
        for u in self.all_users():
            user_items = set([item for (_, item) in self.ur[u]])
            anti_testset += [(self.to_raw_uid(u), self.to_raw_iid(i), fill)
                             for i in self.all_items()
                             if i not in user_items]

        return anti_testset

    def all_users(self):
        """Generator function to iterate over all users.

        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.

        Yields:
            Inner id of items.
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([rating for (_, _, rating) in
                                         self.all_ratings()])

        return self._global_mean
