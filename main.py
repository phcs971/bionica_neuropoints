from helpers.screen_helper import ScreenHelper
from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def main():
    df = pd.read_csv("data/chbmit/chb01_chb01_01.csv")
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

    df["FP1-C3"] = df["FP1-F3"] + df["F3-C3"]
    df["FP2-C4"] = df["FP2-F4"] + df["F4-C4"]

    lower = df.quantile(0.005)
    upper = df.quantile(0.995)
    # print(len(df))


    current = 0
    changes = 0
    lastTime = time["timestamp"].iloc[0]
    speed = 60

    simulate = False

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
    plt.show()

    # ScreenHelper.turn_off()
    # time.sleep(3)
    # ScreenHelper.turn_on()


if __name__ == '__main__':
    main()