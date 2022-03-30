from operator import index
import pandas as pd
import numpy as np
import os

from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder

use_col = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width',
           'Species']


def raw_data_process(raw_data_dir, target_dir):
    filename = os.path.join(raw_data_dir, 'iris.csv')
    raw_data = pd.read_csv(filename, usecols=use_col)
    raw_data = shuffle(raw_data)

    label_Species = LabelEncoder()
    label_Species.fit(np.array(raw_data['Species'].unique()))
    raw_data['Species'] = label_Species.transform(raw_data['Species'])

    t_percent = 0.8

    train_data = raw_data.iloc[:int(len(raw_data) * t_percent), :]
    dev_data = raw_data.iloc[int(len(raw_data) * t_percent):, :]

    # TODO:对数据进行标准化等处理

    np.save(os.path.join(target_dir, 'trainingSet_column_names.npy'),
            train_data.columns.to_numpy())
    np.save(os.path.join(target_dir, 'trainingSet.npy'), train_data.to_numpy())
    np.save(os.path.join(target_dir, 'devSet_column_names.npy'),
            dev_data.columns.to_numpy())
    np.save(os.path.join(target_dir, 'devSet.npy'),
            dev_data.to_numpy())


if __name__ == '__main__':
    #     raw_data_dir = "data_batching/raw_data/"
    #     target_dir = "data_batching/clean_data/"
    raw_data_dir = sys.argv[1]
    target_dir = sys.argv[2]
    raw_data_process(raw_data_dir, target_dir)
