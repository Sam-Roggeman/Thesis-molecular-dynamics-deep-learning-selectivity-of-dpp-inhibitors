import configparser

class ConfigParser:
    def __init__(self, config_path="config.ini"):
        self.config = configparser.ConfigParser()
        self.config_path = config_path
        self.config.read(self.config_path)

    def get(self, section, option):
        return self.config.get(section, option)

    def get_section(self, section):
        return dict(self.config.items(section))

    def get_random(self):
        return self.get_section('Random')

    def get_logging(self):
        return self.get_section('Logging')

