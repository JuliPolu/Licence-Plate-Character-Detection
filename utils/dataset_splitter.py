from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from skmultilearn.model_selection.iterative_stratification import IterativeStratification


#Create the one-hot DataFrame based on class labels
def onehot_classes(df_class_list: pd.DataFrame, class_dict: dict) -> pd.DataFrame:
    df_one_hot = pd.DataFrame(np.zeros((len(df_class_list), len(class_dict))), columns=[str(i) for i in range(len(class_dict))])
    df_one_hot.astype('int8')
    df_one_hot.index = df_class_list.index

    for index, row in df_class_list.iterrows():
        for class_label in row['class']:
            class_label_str = str(class_label)
            df_one_hot.at[index, class_label_str] += 1
    df_class_list = pd.concat([df_class_list, df_one_hot], axis=1)
    return df_class_list


def stratify_shuffle_split_subsets(
    annotation: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:

    x_columns = ['filename']
    y_columns = list(annotation.select_dtypes('float').columns)

    all_x = annotation[x_columns].to_numpy()
    all_y = annotation[y_columns].to_numpy()

    train_indexes, valid_indexes = _split(all_x, all_y, distribution=[1 - train_fraction, train_fraction])
    x_train, x_valid = all_x[train_indexes], all_x[valid_indexes]

    x_train = [i[0] for i in x_train]
    x_valid = [i[0] for i in x_valid]

    return x_train, x_valid


def _split(
    xs: np.array,
    ys: np.array,
    distribution: Union[None, List[float]] = None,
) -> Tuple[np.array, np.array]:
    stratifier = IterativeStratification(n_splits=2, sample_distribution_per_fold=distribution)
    first_indexes, second_indexes = next(stratifier.split(X=xs, y=ys))

    return first_indexes, second_indexes
