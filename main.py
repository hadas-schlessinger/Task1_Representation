import os
import numpy as np
import data_preparation as dp
import model


def main():
    params = dp.get_default_parameters()   # (experiment specific parameters override)
    np.random.seed(0)  # Seed
    dand_l = dp.get_data(params['Data'])
    split_data = dp.train_split_data(dand_l['Data'], dand_l['Labels'], params['Split'])
    # returns train data, test data, train labels and test labels
    train_data_rep = dp.prepare(split_data['Train']['Data'], params['Prepare'])
    train_model = model.train_with_tuning(train_data_rep, split_data['Train']['Labels'], params['Train'])
    test_data_rep = dp.prepare(split_data['Test']['Data'], params['Preapare'])
    results = model.test(train_model, test_data_rep)
    summary = model.evaluate(results, split_data['Test']['Labels'], params['Summary'])
    model.report_results(summary, params['Report'])


if __name__ == "__main__":
    main()
