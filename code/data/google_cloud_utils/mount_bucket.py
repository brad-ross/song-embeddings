import os
import time

buckets = ['song-embeddings-raw-previews']

for bucket in buckets:
    bucketPath = os.path.expanduser('~/' + bucket)
    checkPath = bucketPath + '/dirEmptyCheck'
    bucketName = bucket
    try:
        os.listdir(checkPath)
    except OSError:
        os.system('gcsfuse %s %s' % (bucketName, bucketPath))
        os.system('touch %s' % checkPath)
    time.sleep(3)
