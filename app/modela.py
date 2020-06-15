from joblib import load
import pandas as pd
import preprocessing as pre


class ModelA:
    def __init__(self, name):
        print('name', name)
        self.model = load(f'{name}.joblib')

    def predict(self, entries: pd.DataFrame):
        preprocessed = self.preprocess(entries)
        entries['will_make_purchase'] = self.model.predict(preprocessed)
        return entries

    def preprocess(self, entries: pd.DataFrame):
        preprocessed = entries.loc[:, ['total_duration',]]
        return preprocessed
