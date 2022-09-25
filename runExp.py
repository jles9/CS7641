
import util
from DTAgent import DTAgent
from NNAgent import NNAgent
from KNNAgent import KNNAgent
from SVMAgent import SVMAgent
from BoostingAgent import BoostingAgent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import pdb


def runExpDataSet1():
    fname = "Data/winequality-red.csv"
    data = util.get_data(fname)
    X =  data.iloc[:,:-1]

    conditions = [
        (data['y'] <= 5),
        (data['y'] == 6),
        (data['y'] > 6)
        ]
    
    values = [1,2,3]
    Y = np.select(conditions, values)
    

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    runDTExp(x_train, x_test, y_train, y_test, X, Y, dataname="Wine", scorer='accuracy') 
    runNNExp(x_train, x_test, y_train, y_test, X, Y, dataname="Wine", scorer='accuracy') 
    runKNNExp(x_train, x_test, y_train, y_test, X, Y, dataname="Wine", scorer='accuracy') 
    runSVMExp(x_train, x_test, y_train, y_test, X, Y, dataname="Wine", scorer='accuracy') 
    runBoostingExp(x_train, x_test, y_train, y_test, X, Y, dataname="Wine", scorer='accuracy') 


def runExpDataSet2():
    dtAgent = DTAgent(dataset="Water")
    fname = "Data/water_potability.csv"
    data = util.get_data(fname)
    data = data.dropna()
    X =  data.iloc[:,:-1]
    Y = data.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    runDTExp(x_train, x_test, y_train, y_test, X, Y, dataname="Water",scorer='accuracy') 
    runNNExp(x_train, x_test, y_train, y_test, X, Y, dataname="Water",scorer='accuracy') 
    runKNNExp(x_train, x_test, y_train, y_test, X, Y, dataname="Water",scorer='accuracy') 
    runSVMExp(x_train, x_test, y_train, y_test, X, Y, dataname="Water",scorer='accuracy') 
    runBoostingExp(x_train, x_test, y_train, y_test, X, Y, dataname="Water",scorer='accuracy') 

def runDTExp(x_train, x_test, y_train, y_test, X, Y, dataname, scorer):
    print("Running DT Experiment with Dataset:{}".format(dataname))
    dtAgent = DTAgent(dataset=dataname, scorer=scorer)
    dtAgent.initModel(x_train,y_train)

    dtAgent.plot_learning_timing_curve(x_train, y_train)
    dtAgent.plot_validation_curve(x_train, y_train)
    dtAgent.get_cv_results(x_train, y_train)
    dtAgent.get_final_acc(x_test, y_test)
    dtAgent.save_final_params()

def runNNExp(x_train, x_test, y_train, y_test, X, Y, dataname, scorer):
    print("Running NN Experiment with Dataset:{}".format(dataname))
    nnAgent = NNAgent(dataset=dataname, scorer=scorer)
    nnAgent.initModel(x_train,y_train)

    nnAgent.plot_learning_timing_curve(x_train, y_train)
    nnAgent.plot_validation_curve(x_train, y_train)
    nnAgent.get_cv_results(x_train, y_train)
    nnAgent.get_final_acc(x_test, y_test)
    nnAgent.save_final_params()

def runKNNExp(x_train, x_test, y_train, y_test, X, Y, dataname, scorer):
    print("Running KNN Experiment with Dataset:{}".format(dataname))
    knnAgent = KNNAgent(dataset=dataname, scorer=scorer)
    knnAgent.initModel(x_train,y_train)

    knnAgent.plot_learning_timing_curve(x_train, y_train)
    knnAgent.plot_validation_curve(x_train, y_train)
    knnAgent.get_cv_results(x_train, y_train)
    knnAgent.get_final_acc(x_test, y_test)
    knnAgent.save_final_params()

def runSVMExp(x_train, x_test, y_train, y_test, X, Y, dataname, scorer):
    print("Running SVM Experiment with Dataset:{}".format(dataname))
    svmAgent = SVMAgent(dataset=dataname, scorer=scorer)
    svmAgent.initModel(x_train,y_train)

    svmAgent.plot_learning_timing_curve(x_train, y_train)
    svmAgent.plot_validation_curve(x_train, y_train)
    svmAgent.get_cv_results(x_train, y_train)
    svmAgent.get_final_acc(x_test, y_test)
    svmAgent.save_final_params()

def runBoostingExp(x_train, x_test, y_train, y_test, X, Y, dataname, scorer):
    print("Running Boosting Experiment with Dataset:{}".format(dataname))
    boostingAgent = BoostingAgent(dataset=dataname, scorer=scorer)
    boostingAgent.initModel(x_train,y_train)

    boostingAgent.plot_learning_timing_curve(x_train, y_train)
    boostingAgent.plot_validation_curve(x_train, y_train)
    boostingAgent.get_cv_results(x_train, y_train)
    boostingAgent.get_final_acc(x_test, y_test)
    boostingAgent.save_final_params()




def main():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ConvergenceWarning, module="sklearn"
        )
    runExpDataSet1()
    runExpDataSet2()


if __name__ == "__main__":
    main()


'''
Sanity check for how unbalanced this dataset is:
fname = "Data/winequality-white.csv"
data = util.get_data(fname)
data['y'].value_counts()
6    2198
5    1457
7     880
8     175
4     163
3      20
9       5
'''

'''
1. Load dataset with pandas
2. Pre process data
3. Split data
4. Normalize data
5. Feed traininf data tolearner
6. Test learner on test data
7. Plots

'''
