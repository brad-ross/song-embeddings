import os
import time
from ...config import get_config

config = get_config()
mount_location = config['storage']['mount_location']
buckets = config['storage']['buckets_to_mount']

check_folder = 'dirEmptyCheck'

for bucket in buckets:
    print('mounting {}'.format(bucket))
    bucketPath = os.path.expanduser(os.path.join(mount_location, bucket))
    print(bucketPath)
    checkPath = bucketPath + '/dirEmptyCheck'
    bucketName = bucket
    try:
        os.listdir(checkPath)
    except OSError:
        os.system('gcsfuse %s %s' % (bucketName, bucketPath))
        if check_folder not in set(os.listdir(bucketPath)):
            os.system('mkdir %s' % checkPath)
    time.sleep(3)
