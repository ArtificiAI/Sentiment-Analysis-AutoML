import os

DATASET_CSV_FILENAME = "dataset.csv"


class DataLoader:

    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def load_data(self):
        """
        The loaded file must have the following format with y as first column, a header line, and X as remaining columns

            Y,X
            0,I like potatoes
            0,I do really like bacon11!!1!!
            1,No, I don't like potatoes
            1,Nope.
            0,This is awesome, I want more of this, there are many commas in this sentence and I don't care.
            2.This 2nd sentiment is probably nostalgy. It is what you want it to be maybe.
            3.You can even have more sentiments: just change the number at the beginning.
            3.And be sure you have enough data for each sentiment.

        :return: a tuple of (X, y), they are two lists: "X" is a list of string, "y" is a list of their integer labels
        """
        # return pandas.read_csv(os.path.join(self.data_path, DATASET_CSV_FILENAME)).values
        with open(os.path.join(self.data_path, DATASET_CSV_FILENAME)) as f:
            data = f.readlines()

        X = []
        y = []
        # Will skip first header line:
        for line in data[1:]:
            _y, _, _x = line.partition(",")
            X.append(_x)
            y.append(int(_y))

        return X, y
