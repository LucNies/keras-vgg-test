import numpy as np
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt
import config


def load_train_set():
    df = pd.read_csv(config.TRAIN_FILE)
    columns = df.columns.values
    data = df[columns[1:]].values
    labels = df[columns[0]].values
    mean = data.mean(axis=0)

    data = data-mean
    data = data.reshape((data.shape[0],) + config.IMAGE_SHAPE)

    return data, labels, mean


if __name__ == "__main__":
    data, labels, mean = load_train_set()
    embed()