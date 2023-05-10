from helpers.classifier_helper import ClassifierHelper

def main():
    # ClassifierHelper.classify_v1("data/chbmit/chb01_chb01_01.csv")
    # ClassifierHelper.classify_v2("data/eegmmidb/S001_S001R03.csv", simulate=True)
    ClassifierHelper.classify_v2("data/eegmmidb/eyes_open/S001_S001R01.csv", simulate=False)
    ClassifierHelper.classify_v2("data/eegmmidb/eyes_closed/S001_S001R02.csv", simulate=False)
    # ClassifierHelper.train_v2()
    # ScreenHelper.turn_off()
    # time.sleep(3)
    # ScreenHelper.turn_on()


if __name__ == '__main__':
    main()