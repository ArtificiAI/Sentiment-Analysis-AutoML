import os

from joblib import dump, load
from sklearn.base import BaseEstimator

MODEL_FILE_EXTENSION = ".joblib"


class ModelCheckpoint:

    def __init__(self, model_name_for_saving_path: str, model_checkpoint_path: str = "./cache"):
        self.model_name = model_name_for_saving_path
        self.model_checkpoint_path = model_checkpoint_path

        if not os.path.exists(self.model_checkpoint_path):
            os.mkdir(self.model_checkpoint_path)

    def save_model(self, cls: BaseEstimator):
        dump(cls, os.path.join(self.model_checkpoint_path, self.model_name + MODEL_FILE_EXTENSION))

    def load_model(self) -> BaseEstimator:
        return load(os.path.join(self.model_checkpoint_path, self.model_name + MODEL_FILE_EXTENSION))
