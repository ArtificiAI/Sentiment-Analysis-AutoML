import os
import shutil

from sklearn.linear_model import LinearRegression

from sentiment_analysis_auto_ml.model_checkpoint import ModelCheckpoint

TESTING_CACHE_DIR = "./testing_cache"


def test_can_checkpoint():
    if os.path.exists(TESTING_CACHE_DIR):
        shutil.rmtree(TESTING_CACHE_DIR)
    try:
        mc = ModelCheckpoint("linreg_test_01", model_checkpoint_path=TESTING_CACHE_DIR)

        mc.save_model(LinearRegression())
        p = mc.load_model()

        assert isinstance(p, LinearRegression)
    finally:
        shutil.rmtree(TESTING_CACHE_DIR)
