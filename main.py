import os
import numpy as np
import data_preparation as dp
import model


def main():
    data_path = os.path.join(os.getcwd(), '101_ObjectCategories')
    class_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    class_tuning = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    params = dp.get_default_parameters(data_path, class_indices)   # (experiment specific parameters override)
    np.random.seed(0)  # Seed
    pickle_file = dp.get_data(params)
    split_data = dp.split_data(pickle_file, params['Split'])
    # # returns train data, test data, train labels and test labels
    # train_data_rep = dp.prepare(split_data['Train']['Data'], params['Prepare'])
    # tuning_parameters = model.tuning(class_tuning)
    # train_model = model.train(train_data_rep, split_data['Train']['Labels'], params['Train'], class_indices, tuning_parameters)
    # test_data_rep = dp.prepare(split_data['Test']['Data'], params['Preapare'])
    # results = model.test(train_model, test_data_rep)
    # summary = model.evaluate(results, split_data['Test']['Labels'], params['Summary'])
    # model.report_results(summary, params['Report'])


if __name__ == "__main__":
    main()
