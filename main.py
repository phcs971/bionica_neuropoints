from helpers.screen_helper import ScreenHelper
import time

def main():
    ScreenHelper.turn_off()
    time.sleep(3)
    ScreenHelper.turn_on()


if __name__ == '__main__':
    main()