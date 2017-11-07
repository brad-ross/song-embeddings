from .google_cloud_utils import mount_bucket
import shutil

def save_raw_preview(raw_track, id):
    abs_path = os.path.expanduser('~/song-embeddings-raw-previews/')
    with open(abs_path + '/{0}.mp3').format(id), 'wb') as out_file:
        shutil.copyfileobj(raw_track, out_file)