from config import get_config
import requests
from base64 import b64encode
from time import sleep
from utils import log

MAX_RETRIES = 3
RETRY_DELAY = 5

class Client(object):
    def __init__(self):
        config = get_config()
        
        self.__client_id = config['spotify_auth']['client_id']
        self.__client_secret = config['spotify_auth']['client_secret']
        
        self.__refresh_access_token()

    def __refresh_access_token(self):
        self.__access_token = self.__get_access_token()

    def __get_access_token(self):
        encoded_auth_info = b64encode('{id}:{secret}'.format(id=self.__client_id, 
                                                             secret=self.__client_secret)
                                                     .encode('ascii')).decode()
        auth_req = requests.post('https://accounts.spotify.com/api/token',
                          data={'grant_type': 'client_credentials'},
                          headers={'Authorization': 'Basic {0}'.format(encoded_auth_info)})
        
        return auth_req.json()['access_token']

    def request(self, url, params={}, delay=0):
        if delay > 0:
            sleep(delay)

        for retry_num in range(MAX_RETRIES):
            res = requests.get(url,
                               params=params,
                               headers={'Authorization': 'Bearer {0}'.format(self.__access_token)})
            
            if res.status_code == requests.codes.unauthorized:
                self.__refresh_access_token()
            elif res.status_code == requests.codes.ok:
                return res.json()
            else:
                res_text = res.json() if 'json' in dir(res) else res
                # TODO: use python logger
                log('SPOTIFY ERROR: {0} {1} {2}'.format(url, params, res_text))
            
            sleep(RETRY_DELAY)