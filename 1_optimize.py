import os

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

from sklearn.model_selection import train_test_split

from sentiment_analysis_auto_ml.data_loader import DataLoader
from sentiment_analysis_auto_ml.evaluation.cross_validation import get_fitted_best_classifier_from_cross_validation
from sentiment_analysis_auto_ml.model_checkpoint import ModelCheckpoint

from sentiment_analysis_auto_ml.pipeline_factory import NewLogisticPipelineFunctor, \
    PIPELINE_DEFAULT_NAME, get_generic_hyperparams_grid

if __name__ == "__main__":
    mc = ModelCheckpoint(PIPELINE_DEFAULT_NAME)
    dl = DataLoader()
    X, y = dl.load_data()

    # TODO: train-test split to evaluate model.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    best = get_fitted_best_classifier_from_cross_validation(
        get_generic_hyperparams_grid(),
        NewLogisticPipelineFunctor(),
        X_train, y_train,
        name=PIPELINE_DEFAULT_NAME,
        verbose=True
    )
    score = best.score(X_train, y_train)
    print("Train score:", score)
    score = best.score(X_test, y_test)
    print("Test score:", score)

    # Re-fit on complete data:
    best = NewLogisticPipelineFunctor()().set_params(**best.get_params()).fit(X, y)
    score = best.score(X, y)
    print("Ful-data retrain score:", score)

    model = mc.save_model(best)
