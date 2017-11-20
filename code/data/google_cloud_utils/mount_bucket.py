import os
import time
from ...config import get_config

config = get_config()
mount_location = config['storage']['mount_location']
buckets = config['storage']['buckets_to_mount']

check_file = 'dirEmptyCheck'

for bucket in buckets:
    print('mounting {}'.format(bucket))
    bucketPath = os.path.expanduser(os.path.join(mount_location, bucket))
    checkPath = os.path.join(bucketPath, check_file)
    bucketName = bucket
    if not os.path.isfile(checkPath):
        os.system('gcsfuse %s %s' % (bucketName, bucketPath))
        if not os.path.isfile(checkPath):
            os.system('touch %s' % checkPath)
    	time.sleep(3)
