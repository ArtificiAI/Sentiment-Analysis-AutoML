from sentiment_analysis_auto_ml.data_loader import DataLoader
from sentiment_analysis_auto_ml.evaluation.cross_validation import get_fitted_best_classifier_from_cross_validation
from sentiment_analysis_auto_ml.model_checkpoint import ModelCheckpoint

from sentiment_analysis_auto_ml.pipeline_factory import get_generic_hyperparams_grid, NewLogisticPipelineFunctor, PIPELINE_DEFAULT_NAME, get_test_hyperparams

if __name__ == "__main__":
    mc = ModelCheckpoint(PIPELINE_DEFAULT_NAME)
    dl = DataLoader()
    X_train, y_train = dl.load_data()

    # TODO: train-test split to evaluate model.

    # best = get_fitted_best_classifier_from_cross_validation(
    #     get_generic_hyperparams_grid(),
    #     NewLogisticPipelineFunctor(),
    #     X_train, y_train,
    #     name=name, verbose=True
    # )
    best = NewLogisticPipelineFunctor()().set_params(**get_test_hyperparams()).fit(X_train, y_train)

    model = mc.save_model(best)
