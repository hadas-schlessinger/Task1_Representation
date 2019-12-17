import os
import numpy as np
import data_presentation as dp
import model

def main():
    Params = dp.get_default_parameters()   # (experiment specific parameters override)
    np.random.seed(0); # Seed
    DandL = dp.get_data(Params[‘Data’])
    SplitData = dp.train_split_data(DandL['Data'], DandL['Labels'], Params['Split']) # returns train data, test data, train labels and test labels
    TrainDataRep = dp.prepare(SplitData['Train']['Data'], Params['Prepare'])
    Model = train(TrainDataRep, SplitData['Train']['Labels'], Params['Train'])
    TestDataRep = Prepare(SplitData['Test']['Data'], Params['Preapare'])
    Results = Test(Model, TestDataRep)
    Summary = Evaluate(Results, SplitData['Test']['Labels'], Params['Summary'])
    ReportResults(Summary, Params['Report'])



    if __name__ == "__main__":
    main()
