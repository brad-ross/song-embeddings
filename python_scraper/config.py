import json

def get_config():
    with open('config.json') as config_file:    
        return json.load(config_file)