from config import get_config
from utils import log
from spotify import Client
from db import get_session
from multiprocessing import Pool, Value
from models import Track, Playlist, Album, Artist, Genre

config = get_config()
base_url = 'https://api.spotify.com/v1'
default_country = 'US'
num_categories = 50
rate_limit_delay = 0.1
cats_to_exclude = set(config['scraping']['categories_to_exclude'])

def flatten(iterable):
    return [j for i in iterable for j in i]

def dedupe(objects, attrs_to_concat=set()):
    object_lookup = {}
    for o in objects:
        if o.id not in object_lookup:
            object_lookup[o.id] = o
        elif attrs_to_concat:
            for attr in attrs_to_concat:
                getattr(object_lookup[o.id], attr).extend(getattr(o, attr))

    return object_lookup

def refresh_relationships(objects, relationships):
    for o in objects:
        for r_name, r_objs in relationships.items():
            if isinstance(getattr(o, r_name), list):
                setattr(o, r_name, [r_objs[old_o.id] for old_o in getattr(o, r_name)])
            else:
                setattr(o, r_name, r_objs[getattr(o, r_name).id])

    return objects;

def init_process(counter):
    # TODO: make this nicer
    global client
    client = Client()

    global progress_counter
    progress_counter = counter

def get_playlists_for_category(category_playlists_url):
    res = client.request(category_playlists_url, {
        'limit': num_categories,
        'country': default_country
    })
    return [Playlist.init_from_api(p) for p in res['playlists']['items']]

def get_objects_from_playlist(playlist):
    playlist_tracks_url = base_url + '/users/{user_id}/playlists/{playlist_id}/tracks'.format(
        user_id=playlist.user_id,
        playlist_id=playlist.id
    )
    tracks_res = client.request(playlist_tracks_url)
    raw_tracks = list(map(lambda t: t['track'], tracks_res['items']))

    tracks = []
    albums = []
    artists = []
    for t in raw_tracks:
        track = Track.init_from_api(t)
        tracks.append(track)

        track_artists = [Artist.init_from_api(a) for a in t['artists']]
        track.artists = track_artists
        artists.extend(track_artists)

        track_album = Album.init_from_api(t['album'])
        track.album = track_album
        albums.append(track_album)

        track_album_artists = [Artist.init_from_api(a) for a in t['album']['artists']]
        track_album.artists = track_album_artists
        artists.extend(track_album_artists)

    return playlist.id, tracks, albums, artists

def get_related_artists(artist):
    related_artists_url = base_url + '/artists/{id}/related-artists'.format(id=artist.id)
    related_artists_res = client.request(related_artists_url)

    return [Artist.init_from_api(a) for a in related_artists_res['artists']]

def get_relation_edges(artist):
    if progress_counter.value > 0 and progress_counter.value % 1000 == 0:
        log('{0} artists scraped for related artists'.format(progress_counter.value))

    related_artists_url = base_url + '/artists/{id}/related-artists'.format(id=artist.id)
    related_artists_res = client.request(related_artists_url, delay=rate_limit_delay)

    progress_counter.value += 1

    return artist.id, {artist['id'] for artist in related_artists_res['artists']}

class Scraper(object):
    def __init__(self):
        self.__client = Client()
        self.__db = get_session()
        self.__counter = Value('i', 0)
        self.__pool = Pool(processes=4, initializer=init_process, initargs=(self.__counter,))

    def __parallel_map(self, fn, data):
        results = self.__pool.map(fn, data)
        self.__counter.value = 0
        return results

    def get_public_playlists(self):
        log('Started fetching public playlists...')

        res = self.__client.request(base_url + '/browse/categories', {'limit': num_categories})
        categories_to_scrape = filter(lambda c: c['name'] not in cats_to_exclude, 
                                      res['categories']['items'])
        category_playlists_urls = map(lambda c: c['href'] + '/playlists', categories_to_scrape)

        playlists = flatten(self.__pool.map(get_playlists_for_category, category_playlists_urls))
        deduped_playlists = dedupe(playlists)
        self.__db.add_all(deduped_playlists.values())
        self.__db.commit()

        log('Finished fetching public playlists')

    def get_objects_from_playlists(self):
        log('Started fetching objects from public playlists...')

        playlists = self.__db.query(Playlist).all()
        playlist_lookup = {p.id: p for p in playlists}
        
        playlist_objs = self.__pool.map(get_objects_from_playlist, playlists)
        playlist_track_ref = {p_id: {t.id for t in tracks} for (p_id, tracks, _, __) in playlist_objs}

        log('Finished fetching objects from Spotify API...')

        raw_p, raw_t, raw_alb, raw_art = zip(*playlist_objs)
        tracks, albums, artists = flatten(raw_t), flatten(raw_alb), flatten(raw_art)

        log('raw counts: {0} tracks, {1} albums, {2} artists'.format(
            len(tracks), 
            len(albums), 
            len(artists)
        ))

        deduped_tracks = dedupe(tracks)
        deduped_albums = dedupe(albums)
        deduped_artists = dedupe(artists)

        log('Finished deduplicating objects...')
        log('Unique counts: {0} tracks, {1} albums, {2} artists'.format(
            len(deduped_tracks), 
            len(deduped_albums), 
            len(deduped_artists)
        ))

        final_tracks = refresh_relationships(deduped_tracks.values(), {
            'artists': deduped_artists,
            'album': deduped_albums
        })
        final_albums = refresh_relationships(deduped_albums.values(), {'artists': deduped_artists})
        final_artists = deduped_artists.values()

        log('Finished refreshing relationships...')

        self.__db.expunge_all() # to get rid of duplicate objects cached by the db session
        self.__db.add_all(final_tracks)
        self.__db.commit()

        log('Finished committing objects...')

        self.__db.expunge_all() # clear objects before linking tracks to playlists

        log('Started retrieving tracks and playlists for association...')

        tracks_from_db = self.__db.query(Track).all()
        tracks_lookup = {t.id: t for t in tracks_from_db}
        playlists_from_db = self.__db.query(Playlist).all()
        playlist_lookup = {p.id: p for p in playlists_from_db}
        
        for playlist_id, track_ids in playlist_track_ref.items():
            playlist_lookup[playlist_id].tracks = [tracks_lookup[id] for id in track_ids]

        log('Started commiting playlist relationships...')

        self.__db.add_all(playlists_from_db)
        self.__db.commit()

        log('Finished fetching objects from public playlists')

    def build_related_artists_graph(self):
        artists = self.__db.query(Artist).all()
        original_artists = {artist.id for artist in artists}

        log('Starting to scrape artists related to playlist artists...')

        related_artists = flatten(self.__pool.map(get_related_artists, artists))
        deduped_artists = dedupe(artists + related_artists)

        log('{0} unique artists'.format(len(deduped_artists)))

        log('Finished scraping related artists...')

        self.__db.add_all([artist for id, artist in deduped_artists.items() 
                            if id not in original_artists])
        self.__db.commit()

        log('Committed related artists...')

        self.__db.expunge_all()
        
        artists_from_db = self.__db.query(Artist).all()
        artist_lookup = {a.id: a for a in artists_from_db}

        log('{0} artists to scrape for related artists...'.format(len(artists_from_db)))

        related_artist_graph = self.__parallel_map(get_relation_edges, artists_from_db)
        
        log('Finished fetching artist relations...')

        for i in range(len(related_artist_graph)):
            if i > 0 and i % 1000 == 0:
                log('{0} of {1} artist relations created'.format(i, len(related_artist_graph)))

            artist_id, related_artist_ids = related_artist_graph[i]
            related_artists = [artist_lookup[id] for id in related_artist_ids 
                                if id in artist_lookup]
            artist_lookup[artist_id].related_artists = related_artists

        log('Finished building graph...')
        
        self.__db.commit()

        log('Finished committing artist graph')