from scipy.stats import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def train_random_forest(
        X_train,
        y_train,
        cv_n_iter=100,
        cv_n_splits=10,
        model_param_distributions=None,
        random_seed=42,
        n_jobs=-1,
        verbose=1,
):
    if model_param_distributions is None:
        model_param_distributions = {
            "max_depth": stats.randint(2, 100),
            "n_estimators": stats.randint(10, 1000),
        }
    model = RandomForestRegressor(random_state=random_seed)
    cv_obj = RandomizedSearchCV(
        model,
        param_distributions=model_param_distributions,
        n_iter=cv_n_iter,
        cv=TimeSeriesSplit(n_splits=cv_n_splits),
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    # todo: ravel?
    cv_obj.fit(X_train, y_train.ravel())
    model = cv_obj.best_estimator_
    return model
