import os
import time

buckets = ['song-embeddings-orchestral']

for bucket in buckets:
    print('mounting {}'.format(bucket))
    bucketPath = os.path.expanduser('~/' + bucket)
    checkPath = bucketPath + '/dirEmptyCheck'
    bucketName = bucket
    try:
        os.listdir(checkPath)
    except OSError:
        os.system('gcsfuse %s %s' % (bucketName, bucketPath))
        os.system('mkdir %s' % checkPath)
    time.sleep(3)
