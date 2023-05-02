import platform

class PlatformHelper:
    @staticmethod
    def is_mac():
        return platform.system().lower() == "darwin"