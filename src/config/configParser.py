"""Small configuration-file wrapper used by preprocessing and training scripts."""

import configparser

class ConfigParser:
    """Read project INI sections and expose them as strings or dictionaries."""

    def __init__(self, config_path="config.ini"):
        """Load configuration values from ``config_path`` if it exists."""
        self.config = configparser.ConfigParser()
        self.config_path = config_path
        self.config.read(self.config_path)

    def get(self, section, option):
        """Return one option value from a named section."""
        return self.config.get(section, option)

    def get_section(self, section):
        """Return all options in a section as a dictionary."""
        return dict(self.config.items(section))

    def get_random(self):
        return self.get_section('Random')

    def get_logging(self):
        return self.get_section('Logging')

class ConfigParserWrapper(ConfigParser):
    """Expose project-specific convenience accessors over ``ConfigParser``."""

    def get_raw_data_folder(self):
        """Return the configured raw-data folder."""
        return self.get('Raw Data', 'Folder')

    def get_logging(self):
        """Return logging folder, file-enable, and console-enable settings."""

        return self.get('Logging', 'Folder'), self.get('Logging', 'Enable Logging'), self.get('Logging', 'Enable Console')

