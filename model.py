
def _train(train_data_rep, split_data, params):
    # the functions implementing the actual learning algorithm and the classifier
    pass


def _tuning():
    # If hyper parameter tuning is required, replace the call to train() in main,
    # with a call to TrainWithTuning() which internally also does the hyper parameter tuning (see slides 11-14)
    pass


def train_with_tuning(train_data_rep, split_data, params):
    _tuning()
    _train()
    pass


def test(trained_model, test_data_rep):
    pass


def evaluate():
    pass


def report_results():
    pass
