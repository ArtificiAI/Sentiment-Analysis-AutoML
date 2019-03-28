from sentiment_analysis_auto_ml.data_loader import DataLoader


def test_can_load_data():
    dl = DataLoader()

    X, y = dl.load_data()

    assert len(X) != 0
    assert len(X) == len(y)
