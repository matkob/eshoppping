import pandas as pd
from modelb import ModelB
from modela import ModelA
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
        if model_type == 'B':
            model[model_type] = ModelB(name)
        elif model_type == 'A':
            model[model_type] = ModelA(name)
        else:
            raise Exception(f'Model {model_type} not implemented')


def make_prediction(entries: pd.DataFrame):
    entries_A, entries_B = split_entries_to_AB(entries)
    results_A = model['A'].predict(entries_A) if len(entries_A) else entries_A
    results_B = model['B'].predict(entries_B) if len(entries_B) else entries_B
    return pd.concat([results_A, results_B])

def split_entries_to_AB(entries: pd.DataFrame):
    return entries.loc[entries['session_id'] % 2 == 0], entries.loc[entries['session_id'] % 2 == 1]
