# -*- coding: utf-8 -*-
# gml/train/utils.py

import optuna

# ============================================================
# OPTUNA
# ============================================================


def load_best_params_from_optuna(storage_url: str, study_name: str):
    study = optuna.load_study(storage=storage_url, study_name=study_name)
    return study.best_trial.params
