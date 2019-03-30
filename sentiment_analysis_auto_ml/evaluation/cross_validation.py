# This file was copied from:
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

import multiprocessing

from sklearn.model_selection import GridSearchCV


def cross_validate(new_pipeline_funct, hyperparams_grid, X_train, y_train, name, n_folds=4, verbose=True):
    """
    Return the best hyperparameters.
    """
    print("Cross-Validation Grid Search for: '{}'...".format(name))

    pipeline = new_pipeline_funct()

    count = min(4, multiprocessing.cpu_count())
    grid_search = GridSearchCV(
        pipeline, hyperparams_grid, iid=True, cv=n_folds, return_train_score=False, verbose=verbose, scoring="accuracy",
        n_jobs=count, pre_dispatch=count * 2
    )
    grid_search.fit(X_train, y_train)

    if verbose:
        print("Best hyperparameters for '{}' ({}-folds cross validation accuracy score={}):".format(
            name, n_folds, grid_search.best_score_))
    best_params = grid_search.best_params_
    return best_params


def get_fitted_best_classifier_from_cross_validation(
        hyperparams_grid, new_pipeline_funct, X_train, y_train, name, verbose=True
):
    """
    Return a new `new_pipeline_funct` trained on the data with the best hyperparameters.
    """
    best_params = cross_validate(new_pipeline_funct, hyperparams_grid, X_train, y_train, name, verbose=verbose)
    if verbose:
        print(best_params)
        print("")

    best_pipeline = new_pipeline_funct()
    best_pipeline.set_params(
        **best_params
    )

    best_pipeline.fit(X_train, y_train)
    return best_pipeline
