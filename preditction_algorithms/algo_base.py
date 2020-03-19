"""
defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""


class AlgoBase:
    def __init__(self):
        pass

    def fit(self, trainset):
        """Train an algorithm on a given training set.

                This method is called by every derived class as the first basic step
                for training an algorithm. It basically just initializes some internal
                structures and set the self.trainset attribute.

                Args:
                    trainset(:obj:`Trainset <surprise.Trainset>`) : A training
                        set, as returned by the :meth:`folds
                        <surprise.dataset.Dataset.folds>` method.

                Returns:
                    self
                """

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """
        :param uid: user id.
        :param iid: item id.
        :param r_ui: true rating
        :param clip: whether to clip the estimator into the rating scale.
        :param verbose: whether to print details of the prediction.
        :return: A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object
            containing:

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
            - Some additional details about the prediction that might be useful
              for later analysis.
        """
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)

        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

