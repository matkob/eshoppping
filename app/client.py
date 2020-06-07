import requests
import pandas as pd
import atexit
import random
import base64
import pickle
from apscheduler.schedulers.background import BackgroundScheduler


features = pd.read_pickle('../data/features_prod.pickle')
y_columns = ['delivery_duration', 'syntetic_delivery_duration', 'products_bought', 'made_purchase']


def send_prediction_request():
    random_entries = features.sample(n=random.randint(1, 10))
    x_entries = random_entries.drop(y_columns, axis=1)
    resp = requests.post('http://localhost:8080/prediction', data=base64.b64encode(pickle.dumps(x_entries)))
    prediction = pickle.loads(base64.b64decode(resp.content))
    prediction['made_purchase'] = random_entries.made_purchase
    score = (prediction.made_purchase == prediction.will_make_purchase).sum() / len(prediction.will_make_purchase)
    print(f'predicted {len(random_entries)} entries with score of {round(score, 4)}')


def initialize(request_interval_sec):
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=send_prediction_request, trigger='interval', seconds=request_interval_sec)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
