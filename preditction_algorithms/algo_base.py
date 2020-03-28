"""
defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""

from surprise import similarities as sims

from surprise.prediction_algorithms.predictions import PredictionImpossible
from surprise.prediction_algorithms.predictions import Prediction


class AlgoBase:
    def __init__(self, **kwargs):
        self.bal_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True

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

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # if the details dict was also returned.
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()

            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(iuid, iiid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        """Used when the ``PredictionImpossible`` exception is raised during a
        call to :meth:`predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
        default, return the global mean of all ratings (can be overridden in
        child classes).

        Returns:
            (float): The mean of all ratings in the trainset.
        """
        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        predictions = [self.predict(uid, iid, r_ui_trans, clip=True, verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]

        return predictions

    def compute_baselines(self):
        pass

    def compute_similarities(self):
        """
        Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        :return: The similarity matrix.
        """
        construction_func = {'cosine': sims.cosine,
                             'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        print("ignore some content")

        try:
            if getattr(self, 'verbose', False):
                print("Computing the {0} similarity matrix".format(name))
            sim = construction_func[name](*args)
            if getattr(self, 'verbose', False):
                print("Done computing similarity matrix.")
            return sim
        except KeyError:
            raise NameError("Wrong sim name " + name + ". Allowed values are "
                            ", ".join(construction_func.keys()) + ".")

    def get_neighbors(self, iid, k):
        pass
