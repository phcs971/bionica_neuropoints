import screen_brightness_control as sbc
from helpers.platform_helper import PlatformHelper
import os

class ScreenHelper:
    @staticmethod
    def _mac_set_brightness(value: int):
        result = os.system(f"brightness {value}")
        if (result != 0):
            raise Exception("Please install brightness:\n\nbrew install brightness")
        
    @staticmethod
    def _win_lin_set_brightness(value: int):
        sbc.set_brightness(value)

    @staticmethod
    def turn_off():
        if (PlatformHelper.is_mac()):
            ScreenHelper._mac_set_brightness(0.1)
        else:
            ScreenHelper._win_lin_set_brightness(0)

    @staticmethod
    def turn_on():
        if (PlatformHelper.is_mac()):
            ScreenHelper._mac_set_brightness(1)
        else:
            ScreenHelper._win_lin_set_brightness(100)