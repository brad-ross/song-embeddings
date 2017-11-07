import os
import json

def get_config():
    with open(os.path.join(os.path.dirname(__file__), 'config.json')) as config_file:    
        return json.load(config_file)