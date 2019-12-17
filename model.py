
def _train(train_data_rep, split_data, params):
    # the functions implementing the actual learning algorithm and the classifier
    pass


def _tuning():
    # If hyper parameter tuning is required, replace the call to train() in main,
    # with a call to TrainWithTuning() which internally also does the hyper parameter tuning (see slides 11-14)
    pass


def train_with_tuning(train_data_rep, split_data, params):
    _tuning()
    _train(train_data_rep, split_data, params)
    pass


def test(trained_model, test_data_rep):
    pass


def evaluate(results,split_data, params):
    # Compute the results statistics and return them as fields of Summary For classification these are:
    # Most important: the error rate In our case also:
    # Confusion matrix, the indices of the largest error images
    pass


def report_results(summary, params):
    # Draws the results figures, reports results to the screen
    # Saves the results to the results path, to a file named according to the experiment name or number (e.g. to Results\ResultsOfExp_xx.pkl)
    pass
