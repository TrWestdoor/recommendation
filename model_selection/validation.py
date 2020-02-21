from .split import get_cv


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None):
    measures = [m.lower() for m in measures]

    cv = get_cv(cv)


