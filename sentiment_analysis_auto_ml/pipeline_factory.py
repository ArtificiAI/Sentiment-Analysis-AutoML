# This file was adapted from:
#     https://github.com/guillaume-chevalier/Sentiment-Classification-and-Language-Detection
#
# BSD 3-Clause License
#
# Copyright (c) 2018, Guillaume Chevalier
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from typing import Dict, Any

from artifici_lda.logic.stemmer import Stemmer, FRENCH
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_analysis_auto_ml.pipeline_steps.to_lower_case import ToLowerCase
from sentiment_analysis_auto_ml.pipeline_steps.word_tokenizer import WordTokenizer
from sentiment_analysis_auto_ml.utils import identity


def get_generic_hyperparams_grid():
    d = {
        'porter_stemmer__language': [FRENCH],
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_df': [0.9],  # [0.9, 0.95, 0.98, 0.99],
        'count_vect_that_remove_unfrequent_words_and_stopwords__min_df': [1],  # [1, 2],
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_features': [500000],
        'count_vect_that_remove_unfrequent_words_and_stopwords__ngram_range': [(1, 3)],  # [(1, 2), (1, 3), (1, 4)],
        'count_vect_that_remove_unfrequent_words_and_stopwords__strip_accents': [None],
        'count_vect_that_remove_unfrequent_words_and_stopwords__tokenizer': [identity],
        'count_vect_that_remove_unfrequent_words_and_stopwords__preprocessor': [None],
        'count_vect_that_remove_unfrequent_words_and_stopwords__lowercase': [False],
        'logistic_regression__C': [1.0]  # [0.8, 0.9, 0.95, 1.0, 1.1] # [1e-2, 1e-1, 1.0, 1e1, 1e2],
    }

    return d


def get_small_testing_hyperparams_grid():
    d = {
        'porter_stemmer__language': [FRENCH],
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_df': [0.9, 0.98],
        'count_vect_that_remove_unfrequent_words_and_stopwords__min_df': [1, 2],
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_features': [500000],
        'count_vect_that_remove_unfrequent_words_and_stopwords__ngram_range': [(1, 2)],
        'count_vect_that_remove_unfrequent_words_and_stopwords__strip_accents': [None],
        'count_vect_that_remove_unfrequent_words_and_stopwords__tokenizer': [identity],
        'count_vect_that_remove_unfrequent_words_and_stopwords__preprocessor': [None],
        'count_vect_that_remove_unfrequent_words_and_stopwords__lowercase': [False],
        'logistic_regression__C': [1.0, 1.1],
    }
    return d


def get_small_testing_hyperparams() -> Dict[str, Any]:
    d = {
        'porter_stemmer__language': FRENCH,
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_df': 0.9,
        'count_vect_that_remove_unfrequent_words_and_stopwords__min_df': 2,
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_features': 5000,
        'count_vect_that_remove_unfrequent_words_and_stopwords__ngram_range': (1, 2),
        'count_vect_that_remove_unfrequent_words_and_stopwords__strip_accents': None,
        'count_vect_that_remove_unfrequent_words_and_stopwords__tokenizer': identity,
        'count_vect_that_remove_unfrequent_words_and_stopwords__preprocessor': None,
        'count_vect_that_remove_unfrequent_words_and_stopwords__lowercase': False,
        'logistic_regression__C': 1.0,
    }
    return d


class NewLogisticPipelineFunctor:
    """Calling this object once intanciated will return a pipeline."""

    def __init__(self):
        pass

    def __call__(self) -> Pipeline:
        steps = []

        # Transformers
        steps += [('porter_stemmer', Stemmer())]
        steps += [('tokenizer', WordTokenizer())]
        steps += [('to_lower_case', ToLowerCase())]
        steps += [('count_vect_that_remove_unfrequent_words_and_stopwords', CountVectorizer())]

        # Classifier
        steps += [('logistic_regression', LogisticRegression())]

        return Pipeline(steps)


PIPELINE_DEFAULT_NAME = "v1"
