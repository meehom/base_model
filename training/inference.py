import numpy as np
import pandas as pd
import torch
import sys
from basic_network import MLP

sys.path.append(".")

use_col = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width',
           'Species']


def main(data_set_dir):

    column_names = np.load(
        data_set_dir + '/devSet_column_names.npy', allow_pickle=True)
    test_cases_np = np.load(
        data_set_dir + '/devSet.npy', allow_pickle=True)
    test_cases = pd.DataFrame(test_cases_np, columns=column_names)

    # set parameters
    test_cases_count = 20
    model = MLP(4, 1)
    # load model
    checkpoint = torch.load('result/basic_model/saved_models/0/best.pt')
    model.load_state_dict(checkpoint['state_dict'])

    all_ape = []
    feature_names_x = use_col[:4]
    target_names_y = use_col[4]

    with torch.no_grad():
        for case in range(test_cases_count):
            if case % 10 == 0:
                print('Case:', case, 'out of', test_cases_count)
            factual_case = np.array(
                test_cases.loc[case, feature_names_x], dtype=np.float64)
            ground_truth = np.array(
                test_cases.loc[case, target_names_y], dtype=np.float64)
            factual_case = torch.from_numpy(factual_case).float()
            ground_truth = torch.from_numpy(ground_truth).float()

            predictions = model(factual_case).item()
            case_ground_truth = ground_truth.item()

            all_ape.append(np.abs(predictions - case_ground_truth))

    test_mape = np.mean(all_ape)
    test_median_ape = np.median(all_ape)
    print('Median APE:', test_median_ape)
    print('Mean APE:', test_mape)


if __name__ == "__main__":
    # data_set_dir = "data_batching/clean_data"
    data_set_dir = sys.argv[1]
    main(data_set_dir)
