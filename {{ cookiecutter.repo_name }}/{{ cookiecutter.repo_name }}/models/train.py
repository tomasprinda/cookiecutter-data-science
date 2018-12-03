import logging

import click
import pandas as pd
import numpy as np
from flexp import flexp
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from tputils import csv_dump
import matplotlib.pyplot as plt

# Used due to bug https://github.com/scikit-learn/scikit-learn/issues/12365
# Can be removed in sklearn=0.20.1
from _encoders import OrdinalEncoder


@click.command()
@click.option('--exp', default="exp", help='Experiment folder')
def main(exp):
    flexp.setup("./experiments", exp, with_date=False, loglevel=logging.INFO, log_filename="experiment.log.txt")

    # Load
    logging.info("Loading data")
    df_train = pd.read_csv("data/data_clean_train.csv")
    df_dev = pd.read_csv("data/data_clean_dev.csv")

    # Preprocess
    logging.info("Preprocessing")
    df_x_train, y_train = xy_split(df_train)
    df_x_dev, y_dev = xy_split(df_dev)

    feature_transformer = FeatureTransformer()
    x_train = feature_transformer.fit_transform(df_x_train)
    x_dev = feature_transformer.transform(df_x_dev)
    feature_names = feature_transformer.get_feature_names()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_dev = imputer.transform(x_dev)

    features_to_scale = ["city mpg__", "Year__", "Engine HP__"]
    scaler = FeatureScaler(StandardScaler(), feature_names, features_to_scale)
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)

    # Fit
    logging.info("Fitting")
    # model = RandomForestRegressor(n_estimators=10)
    model = Ridge(fit_intercept=False, normalize=False, alpha=1.)
    model.fit(x_train, y_train)

    # Eval
    logging.info("Evaluating")
    y_train_pred = model.predict(x_train)
    y_dev_pred = model.predict(x_dev)

    eval_rmse(y_train, y_train_pred, y_dev, y_dev_pred)
    # eval_feature_importance(model, feature_names)
    plot_histograms(x_train, feature_names)


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # Ugly but otherwise col_transformer.feature_names() doesn't work
        StandardScaler.get_feature_names = get_empty_feature_names
        FunctionTransformer.get_feature_names = get_empty_feature_names
        OrdinalEncoder.get_feature_names = get_empty_feature_names
        SimpleImputer.get_feature_names = get_empty_feature_names
        RobustScaler.get_feature_names = get_empty_feature_names

        identity = FunctionTransformer(func=lambda x: x, validate=False)
        reciprocal = FunctionTransformer(func=lambda x: 1/x, validate=False)

        self.col_transformer = ColumnTransformer(
            [
                # categorical
                ("Transmission Type", OneHotEncoder(), ["Transmission Type"]),
                ("Vehicle Size", OrdinalEncoder([['Compact', 'Midsize', 'Large']]), ["Vehicle Size"]),

                # numerical
                ("city mpg", reciprocal, ["city mpg"]),
                ("Year", identity, ["Year"]),
                ("Engine HP", identity, ["Engine HP"]),
            ],
            remainder='drop'
        )

    def fit(self, X):
        self.col_transformer.fit(X)
        return self

    def transform(self, X):
        return self.col_transformer.transform(X)

    def get_feature_names(self):
        return self.col_transformer.get_feature_names()


class FeatureScaler(BaseEstimator, TransformerMixin):

    def __init__(self, scaler, feature_names, features_to_scale):
        self.scaler = scaler
        self.feature_names = feature_names
        self.features_to_scale = features_to_scale
        self.feature_ind_to_scale = [i for i, name in enumerate(feature_names) if name in features_to_scale]

        assert len(self.feature_ind_to_scale) == len(self.features_to_scale), \
            "{} {}".format(self.feature_ind_to_scale, self.features_to_scale)

    def fit(self, X):
        self.scaler.fit(X[:, self.feature_ind_to_scale])
        return self

    def transform(self, X):
        X[:, self.feature_ind_to_scale] = self.scaler.transform(X[:, self.feature_ind_to_scale])
        return X


def eval_rmse(y_train, y_train_pred, y_dev, y_dev_pred):
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))

    file = flexp.get_file_path("metrics.csv")
    header = ["metric", "trainset", "devset"]
    row = ["rmse", str(rmse_train), str(rmse_dev)]
    csv_dump([header, row], file)

    logging.info(", ".join(row))


def eval_feature_importance(model, feature_names):
    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    header = ["feature name", "feature importance"]
    file = flexp.get_file_path("feature_importance.csv")
    csv_dump([header] + feature_importance, file)


def plot_histograms(x_train, feature_names):
    for i, feature_name in enumerate(feature_names):
        plt.hist(x_train[:, i])
        plt.title("Histogram {}".format(feature_name))
        plt.savefig(flexp.get_file_path("histogram_{:02d}".format(i)))
        plt.clf()


def xy_split(df):
    """
    :param pd.DataFrame df:
    :return:
    """
    feature_names = [col for col in df.columns if col != "MSRP"]  # All columns except MSRP
    df_x = df[feature_names]
    df_y = df[['MSRP']]

    # df.values extract numpy ndarray from pd.DataFrame
    # ravel() transforms 2D array to 1D array
    return df_x, df_y.values.ravel()


# Ugly but otherwise col_transformer.feature_names() doesn't work
def get_empty_feature_names(x):
    return [""]


if __name__ == "__main__":
    main()
