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
    dp.set_and_split_data(params)
    train_data = dp.prepare(params, params['Pickle']['PickleTrain'])
    model.tuning(train_data)
    train_model = model.train(train_data, params['Train'], class_indices)
    test_data = dp.prepare(params, params['Pickle']['PickleTest'])
    results = model.test(train_model, test_data)
    # summary = model.evaluate(results, split_data['Test']['Labels'], params['Summary'])
    # model.report_results(summary, params['Report'])


if __name__ == "__main__":
    main()
