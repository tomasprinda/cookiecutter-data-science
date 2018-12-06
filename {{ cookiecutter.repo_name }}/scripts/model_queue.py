from baseline import run
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@click.command()
@click.option('--exp', default="exp", help='Experiment folder')
@click.option('--override/--no-override', default=False, help='If override, experiment folder will be overwritten')
@click.option('--dataset_size', "-d", type=click.Choice(cfg.DATASET_SIZES.keys()), default="1k",
              help='Dataset size to use')
def main(exp, override, dataset_size):
def main():
    queue_name = "q2_"
    dataset_size = "100k"
    override = True
    n_jobs = 14

    run(
        queue_name + "rf_100x40_stripAccent", override, dataset_size,
        model=RandomForestClassifier(n_estimators=100, max_depth=40, n_jobs=n_jobs),
        vectorizer=CountVectorizer(preprocessor=lambda x: strip_accents(x).lower())
    )


if __name__ == "__main__":
    main()
