import pandas as pd
import numpy as np


def multi_hot_encoder(s: pd.Series):
    result = pd.DataFrame(index=s.index)
    numpy_data = pd.DataFrame(s.apply(list).tolist()).to_numpy()
    clean = numpy_data[np.logical_not(pd.isna(numpy_data))]
    unique = np.unique(clean)
    for c in unique:
        result[c] = s.apply(lambda r: c in r)
    return result


def encode_multi_hot(df):
    objects = df.select_dtypes(include=['object']).columns
    encoded_objects = []
    for col in objects:
        encoded_objects.append(multi_hot_encoder(df[col]))
    multi_hot = df.drop(objects, axis=1)
    return pd.concat([multi_hot] + encoded_objects, axis=1)


def normalize_numeric(df, mean, std):
    numeric = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric:
        df[col] = (df[col] - mean[col]) / std[col]
    return df


def one_hot_user(df):
    encoded = pd.concat([df.drop('user_id', axis=1), pd.get_dummies(df.user_id)], axis=1)
    encoded.columns = [str(c) for c in encoded.columns]
    return encoded


def best_features(df, features):
    return df[features]


def fix_dimensions(df, features):
    return df.assign(**{str(c): False for c in features if c not in set(df.columns)})


