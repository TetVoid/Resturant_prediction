from pandas import read_csv, DataFrame
from pandas import merge, concat
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class Regression_module:
    def __init__(self):
        self.__model__ = RandomForestRegressor(n_estimators=100, max_features='sqrt')

    def __decoding_quality_predictors__(self, column_name, dataset):
        data = dataset[column_name]
        index_coding = np.zeros((dataset.shape[0], data.max()))
        for i in range(dataset.shape[0]):
            index_coding[i][data[i] - 1] = 1

        axes_names = []
        for i in range(data.max()):
            axes_names.append(column_name + str(i))

        created_frame = DataFrame(data=index_coding, columns=axes_names)
        return created_frame

    def __data_preprocessing__(self, train):
        data = read_csv("data.csv")
        restaurants = read_csv("restaurants.csv")

        dataset = merge(train, data, on=("City", "Date", "IsHoliday"))
        dataset = merge(dataset, restaurants, on=("City"))
        dataset = dataset.drop(["Date"], axis=1)

        restaurants_frame = self.__decoding_quality_predictors__("Restaurant", dataset)
        dataset = dataset.drop(["Restaurant"], axis=1)
        dataset = concat([dataset, restaurants_frame], axis=1)

        city_frame = self.__decoding_quality_predictors__("City", dataset)
        dataset = dataset.drop(["City"], axis=1)
        dataset = concat([dataset, city_frame], axis=1)

        return dataset

    def train(self, path_to_train_data):

        train = read_csv(path_to_train_data)
        dataset = self.__data_preprocessing__(train)


        for item in ["Temperature", "Fuel_Price", "Unemployment"]:
            drop_point = dataset[item].describe()["75%"] + dataset[item].std()
            dataset.loc[dataset[item] > drop_point, item] = drop_point

        answers_data = dataset["Weekly_Sales"]
        train_data = dataset.drop("Weekly_Sales", axis=1)

        self.__X_train__, self.__X_test__, self.__Y_train__, self.__Y_test__ = train_test_split(train_data, answers_data, test_size=0.4)
        self.__model__.fit(self.__X_train__, self.__Y_train__)

    def r2_test(self):
        return r2_score(self.__Y_test__, self.__model__.predict(self.__X_test__))

    def predict(self, data_path):
        data = read_csv(data_path)
        predict = self.__data_preprocessing__(data)

        created_frame = DataFrame(data=self.__model__.predict(predict), columns=["Weekly_Sales"])
        return concat([data, created_frame], axis=1)