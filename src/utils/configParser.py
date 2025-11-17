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

class ConfigParserWrapper(ConfigParser):
    def get_raw_data_folder(self):
        return self.get('Raw Data', 'Folder')

    def get_logging(self):

        return self.get('Logging', 'Folder'), self.get('Logging', 'Enable Logging'), self.get('Logging', 'Enable Console')

