import pandas as pd
import matplotlib.pyplot as plt
import datetime
from helpers.screen_helper import ScreenHelper
from time import sleep
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import numpy as np

model = LinearSVC()

class ClassifierHelper:
    @staticmethod
    def classify_v1(study: str, simulate=False):
        df = pd.read_csv(study)
        df = df.iloc[1:]

        df.dropna()
        df = df.reset_index(drop=True)

        time = df.loc[:, ["'Time and date'"]]
        time.columns = ["timestamp"]

        time["timestamp"] = pd.to_datetime(time["timestamp"], format="'[%H:%M:%S.%f %d/%m/%Y]'")

        time["delta"] = time.loc[:, "timestamp"].diff().dt.total_seconds() * 1000
        time.fillna(0, inplace=True)

        print(time.head())

        df = df[["'C4-P4'", "'F3-C3'", "'FP2-F4'", "'F4-C4'"]]
        df.columns = ["FP1-F3", "F3-C3", "FP2-F4", "F4-C4"]

        rolling = 1

        df["FP1-F3"] = df["FP1-F3"].astype(float).rolling(rolling).mean()
        df["F3-C3"] = df["F3-C3"].astype(float).rolling(rolling).mean()
        df["FP2-F4"] = df["FP2-F4"].astype(float).rolling(rolling).mean()
        df["F4-C4"] = df["F4-C4"].astype(float).rolling(rolling).mean()

        lower = df.quantile(0.005)
        upper = df.quantile(0.995)

        current = 0
        changes = 0
        lastTime = time["timestamp"].iloc[0]
        speed = 60

        for i in range(0, len(df)):
            value = df["FP1-F3"].iloc[i]
            if (value > upper["FP1-F3"]):
                if (current != 1):
                    timestamp = time["timestamp"].iloc[i]
                    delay = (timestamp - lastTime).total_seconds()/speed
                    lastTime = timestamp
                    if (simulate):
                        sleep(delay)
                        ScreenHelper.turn_on()
                    changes += 1
                    current = 1
            elif (value < lower["FP1-F3"]):
                if (current != -1):
                    timestamp = time["timestamp"].iloc[i]
                    delay = (timestamp - lastTime).total_seconds()/speed
                    lastTime = timestamp
                    if (simulate):
                        sleep(delay)
                        ScreenHelper.turn_off()
                    changes += 1
                    current = -1

        print(changes)

            # print time diff from first to last
        print(time["timestamp"].iloc[-1] - time["timestamp"].iloc[0])

        # plot columns
        df.plot(y=["FP1-F3"])

        # plot line at upper and lower
        plt.axhline(y=upper["FP1-F3"], color='r', linestyle='-')
        plt.axhline(y=lower["FP1-F3"], color='r', linestyle='-')

        plt.show()

    @staticmethod
    def _read_study_v2(file: str, open: bool):
        df = pd.read_csv(file)
        df = df.iloc[1:]

        df = df[["'C3..'", "'P3..'", "'O1..'", "'C4..'", "'P4..'", "'O2..'"]]
        df.columns = ["C3", "P3", "O1", "C4", "P4", "O2"]

        df = df[(df["C3"] != 0) & (df["P3"] != 0) & (df["O1"] != 0) & (df["C4"] != 0) & (df["P4"] != 0) & (df["O2"] != 0)]

        df["C3-P3"] = df["C3"].astype(float) - df["P3"].astype(float)
        df["P3-O1"] = df["P3"].astype(float) - df["O1"].astype(float)
        df["C4-P4"] = df["C4"].astype(float) - df["P4"].astype(float)
        df["P4-O2"] = df["P4"].astype(float) - df["O2"].astype(float)

        df = df[["C3-P3", "P3-O1", "C4-P4", "P4-O2"]]

        df["C3-P3-mean"] = df["C3-P3"].astype(float).rolling(100, min_periods=1).mean()
        df["C3-P3-std"] = df["C3-P3"].astype(float).rolling(100, min_periods=1).std()
        df["C3-P3-median"] = df["C3-P3"].astype(float).rolling(100, min_periods=1).median()
        df["C3-P3-sum"] = df["C3-P3"].astype(float).rolling(100, min_periods=1).sum()

        df["P3-O1-mean"] = df["P3-O1"].astype(float).rolling(100, min_periods=1).mean()
        df["P3-O1-std"] = df["P3-O1"].astype(float).rolling(100, min_periods=1).std()
        df["P3-O1-median"] = df["P3-O1"].astype(float).rolling(100, min_periods=1).median()
        df["P3-O1-sum"] = df["P3-O1"].astype(float).rolling(100, min_periods=1).sum()

        df["C4-P4-mean"] = df["C4-P4"].astype(float).rolling(100, min_periods=1).mean()
        df["C4-P4-std"] = df["C4-P4"].astype(float).rolling(100, min_periods=1).std()
        df["C4-P4-median"] = df["C4-P4"].astype(float).rolling(100, min_periods=1).median()
        df["C4-P4-sum"] = df["C4-P4"].astype(float).rolling(100, min_periods=1).sum()

        df["P4-O2-mean"] = df["P4-O2"].astype(float).rolling(100, min_periods=1).mean()
        df["P4-O2-std"] = df["P4-O2"].astype(float).rolling(100, min_periods=1).std()
        df["P4-O2-median"] = df["P4-O2"].astype(float).rolling(100, min_periods=1).median()
        df["P4-O2-sum"] = df["P4-O2"].astype(float).rolling(100, min_periods=1).sum()

        df.fillna(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
        value = 1 if open else 0

        output = pd.DataFrame([value for i in range(len(df.index))], columns=["open"]).astype(float)

        return df, output

    @staticmethod
    def classify_v2(study: str, simulate=False):
        with open(os.path.join("helpers", 'model_classifier_3.pkl') , "rb") as f:
            model = pickle.load(f)
        data, _ = ClassifierHelper._read_study_v2(study, True)
        predict = model.predict(data.values)

        current = 1
        lastTime = 0

        changes = []

        for i in range(0, len(predict)):
            value = predict[i]
            if value != current:    
                if value == 1:
                    if (simulate):
                        print('on')
                        sleep((i - lastTime)/500)
                        ScreenHelper.turn_on()
                    current = 1
                    changes.append([lastTime, i, 1])
                else:
                    if (simulate):
                        print('off')
                        sleep((i - lastTime)/500)
                        ScreenHelper.turn_off()
                    current = 0
                    changes.append([lastTime, i, 0])
                lastTime = i

        print(changes)




        data.plot(y=["C3-P3-mean", "P3-O1-mean", "C4-P4-mean", "P4-O2-mean"], subplots=True)

        plt.show()
        # data.plot(y=["C3-P3-mean", "P3-O1-mean", "C4-P4-mean", "P4-O2-mean"], subplots=True)
        for change in changes[100:150]:
            if change[2] == 0:
                plt.axvspan(change[0], change[1], color='red', alpha=1)
            else:
                plt.axvspan(change[0], change[1], color='blue', alpha=1)
        plt.show()

    @staticmethod
    def train_v2():
        inputs = []
        outputs = []
        eyes_open = [os.path.join("data/eegmmidb/eyes_open", file) for file in os.listdir("data/eegmmidb/eyes_open")] 
        eyes_closed = [os.path.join("data/eegmmidb/eyes_closed", file) for file in os.listdir("data/eegmmidb/eyes_closed")]
        
        take = 20

        for file in eyes_open[0:take]:
            inp, out = ClassifierHelper._read_study_v2(file, True)
            inputs.append(inp)
            outputs.append(out)

        for file in eyes_closed[0:take]:
            inp, out = ClassifierHelper._read_study_v2(file, False)
            inputs.append(inp)
            outputs.append(out)

        inputs = pd.concat(inputs)
        outputs = pd.concat(outputs)

        print(f"Dados coletados!")

        x_train, x_test, y_train, y_test = train_test_split(inputs, outputs["open"], test_size= 0.2, random_state= 42)

        print(f"Iniciando Treinamento...")

        model.fit(x_train.values, y_train.values)

        print(f"Treinamento concluído!")
        print(f"Acurácia: {accuracy_score(y_test.values, model.predict(x_test.values))}")

        with open(os.path.join("helpers", 'model_classifier.pkl'), "wb") as f:
            pickle.dump(model, f)





