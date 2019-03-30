from sentiment_analysis_auto_ml.data_loader import DataLoader
from sentiment_analysis_auto_ml.pipeline_factory import NewLogisticPipelineFunctor, get_small_testing_hyperparams


def test_sentiment_classifier():
    # TODO: custom demo data for the test.
    dl = DataLoader()
    X_train, y_train = dl.load_data()

    best_model = NewLogisticPipelineFunctor()().set_params(**get_small_testing_hyperparams()).fit(X_train, y_train)
    X = ["Je suis un modèle pas content.", "Je suis un modèle très content."]
    y = best_model.predict(X)
    # y = best_model.predict_proba(X)

    expected_y = [0, 1]
    assert y.tolist() == expected_y
