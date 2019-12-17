
def get_default_parameters():

    # Returns a dict containing the default experiment parameters
    # It has several fields, each itself a dict of parameters for the various experiment stages
    # These are ‘Split’, ‘Prepare’, ‘Train’, ‘Summary’, ‘Report’ (according to the needs)
    # Each struct is sent to the relevant function (i.e. Params[‘Train’] is sent to Train(), etc.)
    # Each experiment starts by configuring the experiment parameters:
    # Calling GetDefaultParameters()
    # Changing relevant parameters according to the specific experiments needs
    # Do not keep hidden constants in the code (use parameters to set them)
    pass


def get_data(path):
    # Loads the data and subsets it if required
    # Puts the data in DandL[‘Data’], the labels in DandL[‘Labels’],
    # In our case: Params include a path for the data and sub - setting parameters

    pass


def train_split_data(TrainData, TestData, TrainLabels, TestLabels):
    # Splits the data and labels according to a ratio defined in Params
    # SplitData includes fields: TrainData, TestData, TrainLabels, TestLabels
    pass


def prepare():
    pass


