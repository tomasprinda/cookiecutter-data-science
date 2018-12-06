import logging

import click
from flexp import flexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from src.utils import stopwords, strip_accents
from tputils import Timer
from xgboost import XGBClassifier

from src import cfg
from numpy_related.data import xy_split, load_data
from src.eval import eval_accuracy, eval_feature_importance, best_predictions, eval_classes, eval_time, \
    eval_main_metrics, eval_model_size
from src.store import store_model_params, store_predictions


@click.command()
@click.option('--exp', default="exp", help='Experiment folder')
@click.option('--override/--no-override', default=False, help='If override, experiment folder will be overwritten')
@click.option('--dataset_size', "-d", type=click.Choice(cfg.DATASET_SIZES.keys()), default="1k",
              help='Dataset size to use')
def main(exp, override, dataset_size):
    # Function split to main and run to be able to call it from model_queue.py
    model = LogisticRegression(solver='newton-cg', n_jobs=1)
    run(exp, override, dataset_size, model)


def run(exp, override, dataset_size, model, vectorizer):
    exp = "{}-{}".format(exp, dataset_size)
    flexp.setup("./experiments", exp, with_date=False, loglevel=logging.INFO, override_dir=override)
    flexp.describe(model.__class__.__name__)

    # Load
    logging.info("Loading")
    df_train, df_dev = load_data(dataset_size, dataset_types=("train", "dev"))

    # Preprocess
    logging.info("Preprocessing")
    ids_train, titles_train, y_train = xy_split(df_train)
    ids_dev, titles_dev, y_dev = xy_split(df_dev)

    # Fit
    logging.info("Fitting")
    model.fit(x_train, y_train)

    # Predict
    logging.info("Predicting")
    with Timer() as t:
        y_train_pred_proba = model.predict_proba(x_train)
        y_train_pred, y_train_pred_proba = best_predictions(y_train_pred_proba, model.classes_, n=10)

        y_dev_pred_proba = model.predict_proba(x_dev)
        y_dev_pred, y_dev_pred_proba = best_predictions(y_dev_pred_proba, model.classes_, n=10)

    # Store
    logging.info("Storing")
    store_model_params(model, flexp.get_file("model_params.json"))
    # store_model(model, flexp.get_file("model_dict.pkl"))
    store_predictions(ids_dev, y_dev_pred, y_dev_pred_proba, dataset_size, flexp.get_file("predictions_dev.npz"))

    # Eval
    logging.info("Evaluating accuracy")
    main_acc = eval_accuracy(y_train, y_train_pred, y_dev, y_dev_pred, flexp.get_file("accuracy.csv"))

    logging.info("Evaluating classes")
    eval_classes(y_dev, y_dev_pred, model.classes_, flexp.get_file("eval_classes.csv"))

    logging.info("Evaluating time")
    n_examples = y_train_pred_proba.shape[0] + y_dev_pred_proba.shape[0]
    main_time = eval_time(t.duration, n_examples, model.n_jobs)

    # logging.info("Evaluating confusion matrix")
    # eval_confusion_matrix(y_dev, y_dev_pred[:, 0], model.classes_, flexp.get_file("cm_dev.png"))

    logging.info("Evaluating feature importance")
    eval_feature_importance(
        model, vectorizer.get_feature_names(), flexp.get_file("feature_importance.csv")
    )

    logging.info("Evaluating model size")
    main_size = eval_model_size(model)

    logging.info("Evaluating main metrics")
    metric_names, metric_values = zip(main_acc, main_time, main_size)
    eval_main_metrics(metric_names, metric_values, flexp.get_file("metrics.csv"))


    flexp.close() # Neccessary to use it in a queue


if __name__ == "__main__":
    main()
