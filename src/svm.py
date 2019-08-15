import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def load_data():

    # load training validation and test data into dataframes
    train_data = pd.read_csv("Training.txt")
    val_data = pd.read_csv("Validation.txt")
    test_data = pd.read_csv("Test.txt")

    # shuffle data and return
    return shuffle(train_data), shuffle(val_data), shuffle(test_data)

def build_data_vectors(train_data, val_data, test_data):

    # build X and y vectors for SVM
    X_train = []
    X_val = []
    X_test = []

    # get temperature feature
    temperatures_train = train_data["Temperature"].values
    temperatures_val = val_data["Temperature"].values
    temperatures_test = test_data["Temperature"].values
    X_train.append(temperatures_train)
    X_val.append(temperatures_val)
    X_test.append(temperatures_test)
    
    # get humidity feature
    humidities_train = train_data["Humidity"].values
    humidities_val= val_data["Humidity"].values
    humidities_test= test_data["Humidity"].values
    X_train.append(humidities_train)
    X_val.append(humidities_val)
    X_test.append(humidities_test)
    
    # get light feature
    lights_train = train_data["Light"].values
    lights_val= val_data["Light"].values
    lights_test= test_data["Light"].values
    X_train.append(lights_train)
    X_val.append(lights_val)
    X_test.append(lights_test)

    # get C02 feature
    co2_train = train_data["CO2"].values
    co2_val= val_data["CO2"].values
    co2_test= test_data["CO2"].values
    X_train.append(co2_train)
    X_val.append(co2_val)
    X_test.append(co2_test)

    # get humidity ratio feature
    humidity_ratios_train = train_data["HumidityRatio"].values
    humidity_ratios_val= val_data["HumidityRatio"].values
    humidity_ratios_test= test_data["HumidityRatio"].values
    X_train.append(humidity_ratios_train)
    X_val.append(humidity_ratios_val)
    X_test.append(humidity_ratios_test)

    # build y vectors
    y_train = train_data["Occupancy"].values
    y_val= val_data["Occupancy"].values
    y_test= test_data["Occupancy"].values

    X_train = np.stack(X_train, axis=1)
    X_val = np.stack(X_val, axis=1)
    X_test = np.stack(X_test, axis=1)

    # return vectors
    return X_train, y_train, X_val, y_val, X_test, y_test

def linear_svc(X_train, y_train, X_val, y_val, C):

    # define SVM with specified penalty and fit on training data
    print("Running Linear SVM with C=" + str(C) + "\n")
    clf = svm.LinearSVC(C=C)
    clf.fit(X_train, y_train)

    # predict y values on validation data
    val_preds = clf.predict(X_val)

    # convert to np arrays
    val_preds = np.asarray(val_preds, dtype=np.int)
    y_val = np.asarray(y_val, dtype=np.int)

    # calculate accuracy
    diff = np.absolute(np.subtract(val_preds, y_val))
    mistakes = np.sum(diff)
    accuracy = ((len(diff) - mistakes)*100) / len(diff)
    print("Accuracy: " + str(accuracy) + "%\n")

if __name__ == "__main__":

    # get shuffled dataframes
    train_data, val_data, test_data = load_data()

    # build X and y vectors for SVMs for training validation and testing
    X_train, y_train, X_val, y_val, X_test, y_test = build_data_vectors(train_data, val_data, test_data)

    # run SVMs with different parameters
    #linear_svc(X_train, y_train, X_val, y_val, 1)
    #linear_svc(X_train, y_train, X_val, y_val, 10)
    #linear_svc(X_train, y_train, X_val, y_val, 100)
    #linear_svc(X_train, y_train, X_val, y_val, 1000)

    # run test set 
    #linear_svc(X_train, y_train, X_test, y_test, 10)

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    score = logistic.score(X_test, y_test)
    print(score)


