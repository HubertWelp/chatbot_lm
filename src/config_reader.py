import configparser
import os
import threading


class ConfigManager:
    """
    A class for managing configurations using configparser, with thread-safe operations.

    Attributes:
        SECTIONS (dict): A dictionary mapping section prefixes to their full section names in the configuration file.
    """

    _lock = threading.RLock()  # reentrant lock
    SECTIONS = {
        "llm": "llm_settings",
        "sweetpicker": "sweetpicker_settings",
    }

    def __init__(self, config_path=None, config_filename="config.ini"):
        """
        Initializes the ConfigManager instance.

        Args:
            config_path (str, optional): The path to the configuration file. Defaults to None.
            config_filename (str, optional): The filename of the configuration file. Defaults to "config.ini".
        """
        # Use the specified path or default to the current directory
        if config_path is None:
            config_path = os.path.dirname(
                os.path.abspath(__file__)
            )  # Directory of the current script

        # Construct the full path to the config file
        self.config_file = os.path.join(config_path, config_filename)
        self.config = configparser.ConfigParser()

        # Check if the config file exists, if not create it with default values
        if not os.path.exists(self.config_file):
            self.create_default_config()

        # Read the configuration file
        self.config.read(self.config_file)
        self._last_timestamp = os.path.getmtime(self.config_file)

    def create_default_config(self):
        """
        Creates a default configuration file with default settings.

        Raises:
            OSError: If the configuration file cannot be written.
        """
        # See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        self.config["llm_settings"] = {
            "model_name": "qwen2.5:14b",
            "systemMessage": "None",
            "keep_alive": "-1",
            "mirostat": "2",
            "mirostat_eta": "0.1",
            "mirostat_tau": "5",
            "num_ctx": "2048",
            "repeat_last_n": "64",
            "repeat_penalty": "1.1",
            "temperature": "0.2",
            "seed": "0",
            "stop": "<|start_header_id|>,<|end_header_id|>,<|eot_id|>,Translated",
            "tfs_z": "1",
            "num_predict": "128",
            "top_k": "40",
            "top_p": "0.7",
        }
        self.config["sweetpicker_settings"] = {
            "items": "Maoam,Snickers,Kinderriegel,Milkyway",
        }

        # Write the default configuration to the file
        with self._lock:
            with open(self.config_file, "w") as configfile:
                self.config.write(configfile)

    def get(self, section, option):
        """
        Retrieves a value from the configuration file.

        Args:
            section (str): The section in the configuration file.
            option (str): The option in the specified section.

        Returns:
            str: The value of the specified option.

        Raises:
            KeyError: If the specified section or option does not exist.
        """
        # Get a value from the config file
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise KeyError(str(e))

    def set(self, section, option, value):
        """
        Sets a value in the configuration file.

        Args:
            section (str): The section in the configuration file.
            option (str): The option in the specified section.
            value (str): The value to set for the specified option.

        Raises:
            KeyError: If the specified section does not exist or the option is not found within the section.
        """
        # Ensure the option exists before setting its value
        if not self.config.has_section(section):
            raise KeyError(f"The section '{section}' does not exist")
        if self.config.has_option(section, option):
            self.config.set(section, option, value)
            # Write changes back to the file with lock
            with self._lock:
                with open(self.config_file, "w") as configfile:
                    self.config.write(configfile)
        else:
            raise KeyError(
                f"The option '{option}' does not exist in section '{section}'"
            )

    def get_setting(self, section_prefix, option):
        """
        Retrieves a value from a specific section using a prefix.

        Args:
            section_prefix (str): The prefix of the section.
            option (str): The option in the specified section.

        Returns:
            str: The value of the specified option.

        Raises:
            KeyError: If the section corresponding to the prefix does not exist.
        """
        section = self.SECTIONS.get(section_prefix)
        if section:
            return self.get(section, option)
        raise KeyError(f"The section '{section_prefix}' does not exist")

    def set_setting(self, section_prefix, option, value):
        """
        Sets a value in a specific section using a prefix.

        Args:
            section_prefix (str): The prefix of the section.
            option (str): The option in the specified section.
            value (str): The value to set for the specified option.

        Raises:
            KeyError: If the section corresponding to the prefix does not exist.
        """
        try:
            section = self.SECTIONS[section_prefix]
            self.set(section, option, value)
        except KeyError:
            raise KeyError(f"The section '{section_prefix}' does not exist")

    def get_all_settings(self, section_prefix) -> dict:
        """
        Retrieves all settings from a specific section using a prefix.

        Args:
            section_prefix (str): The prefix of the section.

        Returns:
            dict: A dictionary containing all settings from the specified section.

        Raises:
            KeyError: If the section corresponding to the prefix does not exist.
        """
        self.reload_if_changed()

        section = self.SECTIONS.get(section_prefix)
        if not section:
            raise KeyError(f"The section '{section_prefix}' does not exist")
        if not self.config.has_section(section):
            raise KeyError(
                f"The section '{section}' does not exist in the configuration"
            )

        return {key: value for key, value in self.config.items(section)}

    def reload_if_changed(self):
        """
        Check if the configuration file has changed and reload it.
        """
        current_timestamp = os.path.getmtime(self.config_file)
        if (
            not hasattr(self, "_last_timestamp")
            or self._last_timestamp != current_timestamp
        ):
            self._last_timestamp = current_timestamp
            self.config.read(self.config_file)
            return True
        return False


config_manager = ConfigManager()

# # Usage example
# config_path = '/path/to/config/directory'  # Replace with your desired config directory path
# config_manager = ConfigManager()

# # Example usage of generalized getters and setters
# print(config_manager.get_setting("chatbot", "ros_topic_user_query"))
# config_manager.set_setting("chatbot", "ros_topic_user_query", "/new/chatbot/query")
# print(config_manager.get_setting("chatbot", "ros_topic_user_query"))

# print(config_manager.get_setting("llm", "model_name"))
# config_manager.set_setting("llm", "model_name", "new-llm-model")
# print(config_manager.get_setting("llm", "model_name"))

# # Get all settings for a section
# print(config_manager.get_all_settings("chatbot"))

# # Attempt to set a non-existent option (this will raise a KeyError)
# try:
#     config_manager.set_setting("chatbot", "non_existent_option", "some_value")
# except KeyError as e:
#     print(e)
