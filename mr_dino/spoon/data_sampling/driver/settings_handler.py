import yaml 

class SettingsHandler:
    """
    Handler for storing a dictionary of settings 
    """
    def __init__(self, settings_dict):
        self._settings = settings_dict
        self._fields = list(self._settings.keys())
    
    def update(self, new_dict):
        self._settings.update(new_dict)

    def get_fields(self):
        """
        Returns a copy of the fields
        """
        return self._fields.copy()

    def get_settings(self):
        """
        Returns a copy of the settings
        """
        return self._settings.copy()
    
    def settings_str(self):
        """
        Puts together the field/values in a long string.

        .. note:: Only prints the original fields given in the settings 
        """
        total_str = '' 
        for i_field, field in enumerate(self._fields):
            if i_field == 0:
                total_str += f'{field}_{self._settings[field]}'
            else:
                total_str += f'_{field}_{self._settings[field]}'
        return total_str

    def make_directory_name(self, pre_path):
        """
        Makes a name for a save directory given as
        :code:`pre_path/settings_str`
        """
        save_dir = f"{pre_path}/{self.settings_str()}"
        return save_dir 

    def update_from_yaml(self, filename):
        if filename is not None:
            print("Load settings from yaml file")
            with open(filename, 'r') as stream:
                loaded_settings = yaml.safe_load(stream)
            self.update(loaded_settings)
        else:
            print("No input file given")

    def update_if_not_none(self, index, value):
        print(index, value)
        if value is not None:
            print("Updating")
            self.__setitem__(index, value)

    def __getitem__(self, key):
        return self._settings[key]

    def __setitem__(self, index, value):
        self._settings[index] = value 

