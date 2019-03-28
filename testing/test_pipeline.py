from sentiment_analysis_auto_ml.data_loader import DataLoader
from sentiment_analysis_auto_ml.pipeline_factory import NewLogisticPipelineFunctor, get_test_hyperparams


def test_sentiment_classifier():
    if __name__ == "__main__":
        # TODO: custom demo data for the test.
        dl = DataLoader()
        X_train, y_train = dl.load_data()

        best_model = NewLogisticPipelineFunctor()().set_params(**get_test_hyperparams()).fit(X_train, y_train)
        X = ["Je suis un modèle pas content.", "Je suis un modèle très content."]
        y = best_model.predict(X)

        expected_y = [0, 1]
        assert y == expected_y
