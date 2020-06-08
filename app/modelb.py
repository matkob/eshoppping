from joblib import load
import pandas as pd
import preprocessing as pre


class ModelB:
    def __init__(self, name):
        self.metadata = pd.read_pickle(f'{name}_metadata.pickle')
        self.data = pd.read_pickle(f'{name}_data.pickle')
        self.model = load(f'{name}.joblib')
        self.y = self.metadata.y
        self.features = self.metadata.features.drop(self.y)
        self.mean = self.data.loc['mean']
        self.std = self.data.loc['std']

    def predict(self, entries: pd.DataFrame):
        preprocessed = self.preprocess(entries)
        entries['will_make_purchase'] = self.model.predict(preprocessed)
        return entries

    def preprocess(self, entries: pd.DataFrame):
        preprocessed = pre.encode_multi_hot(entries)
        preprocessed = pre.one_hot_user(preprocessed)
        preprocessed = pre.normalize_numeric(preprocessed, self.mean, self.std)
        preprocessed = pre.fix_dimensions(preprocessed, self.features)
        return pre.best_features(preprocessed, self.features)
