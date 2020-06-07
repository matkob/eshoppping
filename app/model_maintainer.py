import pandas as pd
from modelb import ModelB
import atexit
from apscheduler.schedulers.background import BackgroundScheduler


model = {}


def initialize(models, reload_interval_min):
    load_models(models)
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=lambda: load_models(models), trigger='interval', minutes=reload_interval_min)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())


def load_models(models):
    print('loading models')
    for model_type, name in models:
        model[model_type] = ModelB(name)


def make_prediction(entries: pd.DataFrame):
    m = choose_model()
    return m.predict(entries)


def choose_model():
    # TODO A/B
    return model['B']

