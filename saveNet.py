"""
    Module:
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2020/12/23
    Introduce: Used to store and load models
"""
__all__ = ["save_model", "load_model"]

import joblib


def save_model(model, path):
    joblib.dump(model, filename=path)


def load_model(path):
    return joblib.load(path)
