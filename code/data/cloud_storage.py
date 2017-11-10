from .google_cloud_utils import mount_bucket
from ..config import get_config
import os
import shutil
from time import sleep

NUM_RETRIES = 5
RETRY_DELAY = 2

config = get_config()
mount_location = config['storage']['mount_location']

def save_raw_preview_to_cloud(raw_track, track_id, bucket_name):
    abs_path = os.path.expanduser(os.path.join(mount_location, bucket_name))
    for i in range(NUM_RETRIES):
	try:
    	    with open(os.path.join(abs_path, '{0}.mp3').format(track_id), 'wb') as out_file:
            	shutil.copyfileobj(raw_track, out_file)
	    return
	except:
	    print('ERROR: error when trying to save track {}'.format(track_id))
	    
        sleep(RETRY_DELAY)
    
    print('ERROR: max retries exceeded when trying to save track {}'.format(track_id))
