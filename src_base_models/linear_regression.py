from sklearn import linear_model


def train_linreg(X_train, y_train, n_jobs=-1):
    model = linear_model.LinearRegression(n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model
