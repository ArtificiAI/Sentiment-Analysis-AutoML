
# Sentiment Analysis AutoML

## Usage

1. Create a `dataset.csv` file under a `data` subolder such as `./data/dataset.csv`.
2. Install requirements: `pip3 install -r requirements.txt`
3. Run `python3 1_optimize.py` to launch AutoML. The best model will be saved under the `.cache/` folder (folder will be created if absent as default).
4. Run `python3 2_serve_main.py` to load the cached model and serve predictions. You could for example build a REST API in this file to serve predictions over the web.

## Format of the dataset

Your `./data/dataset.csv` file needs to look like that:

```
0,I like potatoes
0,I do really like bacon11!!1!!
1,No, I don't like potatoes
1,Nope.
0,This is awesome, I want more of this, there are many commas in this sentence and I don't care.
2,This 2nd sentiment is probably nostalgy. It is what you want it to be maybe.
3,You can even have more sentiments: just change the number at the beginning.
3,And be sure you have enough data for each sentiment.
```

The numbers are what you want them to mean: as long as the label is a number starting from zero. For example, a zero could mean "happy", a one could mean "mad", a two could mean "nostalgy" and a 3 could mean something else. You can have as many numbers as you want. The strings in the CSV file must not be escaped (e.g.: preferably don't use `"` nor `'` characters in the CSV).

## License

This project is published under the [MIT License (MIT)](LICENSE).

Copyright (c) [2018 Artifici online services inc](https://github.com/ArtificiAI).

Coded by [Guillaume Chevalier](https://github.com/guillaume-chevalier) at [Neuraxio Inc.](https://github.com/Neuraxio)
