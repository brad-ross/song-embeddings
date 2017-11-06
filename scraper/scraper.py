from config import get_config
from utils import log
from spotify import Client
from db import get_session
from multiprocessing import Pool, Value
from models import Track, Playlist, Album, Artist, Genre
import shutil

config = get_config()
base_url = 'https://api.spotify.com/v1'
default_country = 'US'
num_categories = 50
rate_limit_delay = 0.2
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
    }, delay=rate_limit_delay)
    return [Playlist.init_from_api(p) for p in res['playlists']['items']]

def get_artists_from_playlist(playlist):
    playlist_tracks_url = base_url + '/users/{user_id}/playlists/{playlist_id}/tracks'.format(
        user_id=playlist.user_id,
        playlist_id=playlist.id
    )
    tracks_res = client.request(playlist_tracks_url, delay=rate_limit_delay)
    raw_tracks = [t['track'] for t in tracks_res['items']]

    return [Artist.init_from_api(a) for t in raw_tracks 
                for a in t['artists'] if t['id'] is not None]

def get_objects_from_raw_tracks(raw_tracks):
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

    return tracks, albums, artists

def get_objects_from_playlist(playlist):
    playlist_tracks_url = base_url + '/users/{user_id}/playlists/{playlist_id}/tracks'.format(
        user_id=playlist.user_id,
        playlist_id=playlist.id
    )
    tracks_res = client.request(playlist_tracks_url, delay=rate_limit_delay)
    raw_tracks = list(map(lambda t: t['track'], tracks_res['items']))

    tracks, albums, artists = get_objects_from_raw_tracks(raw_tracks)

    progress_counter.value += 1

    if progress_counter.value > 0 and progress_counter.value % 100 == 0:
        log('{0} playlists scraped for top tracks'.format(progress_counter.value))

    return playlist.id, tracks, albums, artists

def get_related_artists(artist):
    related_artists_url = base_url + '/artists/{id}/related-artists'.format(id=artist.id)
    related_artists_res = client.request(related_artists_url, delay=rate_limit_delay)

    progress_counter.value += 1

    if progress_counter.value > 0 and progress_counter.value % 1000 == 0:
        log('{0} artists scraped for related artists'.format(progress_counter.value))

    return [Artist.init_from_api(a) for a in related_artists_res['artists']]

def get_relation_edges(artist):
    related_artists_url = base_url + '/artists/{id}/related-artists'.format(id=artist.id)
    related_artists_res = client.request(related_artists_url, delay=rate_limit_delay)

    progress_counter.value += 1

    if progress_counter.value > 0 and progress_counter.value % 1000 == 0:
        log('{0} artists scraped for related artists'.format(progress_counter.value))

    return artist.id, {artist['id'] for artist in related_artists_res['artists']}

def get_top_tracks_from_artist(artist):
    top_tracks_url = base_url + '/artists/{id}/top-tracks'.format(id=artist.id)
    top_tracks_res = client.request(top_tracks_url, {
        'country': default_country
    }, delay=rate_limit_delay)

    progress_counter.value += 1

    if progress_counter.value > 0 and progress_counter.value % 1000 == 0:
        log('{0} artists scraped for top tracks'.format(progress_counter.value))

    tracks, albums, artists = get_objects_from_raw_tracks(top_tracks_res['tracks'])

    for track in tracks:
        track.is_top_track = True

    return tracks, albums, artists

def get_genres_from_object(object):
    object_url = base_url + ('/artists/' if isinstance(object, Artist) else '/albums/') + object.id
    object_res = client.request(object_url, {
        'market': default_country
    }, delay=rate_limit_delay)

    progress_counter.value += 1

    if progress_counter.value > 0 and progress_counter.value % 1000 == 0:
        log('{0} objects scraped for genres'.format(progress_counter.value))

    return object.id, object_res['genres']

def save_raw_preview(track):
    raw_track = client.get_raw_preview(track.preview_url, delay=rate_limit_delay)
    with open('./test_output/{0}.mp3'.format(track.id), 'wb') as out_file:
        shutil.copyfileobj(raw_track, out_file)

    progress_counter.value += 1

    if progress_counter.value > 0 and progress_counter.value % 10 == 0:
        log('{0} raw previews saved'.format(progress_counter.value))

class Scraper(object):
    def __init__(self):
        self.__client = Client()
        log('Spotify client set up')
        self.__db = get_session()
        log('Database session set up')
        self.__counter = Value('i', 0)
        self.__pool = Pool(processes=4, initializer=init_process, initargs=(self.__counter,))
        log('Process pool set up')

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

        playlists = flatten(
            self.__parallel_map(get_playlists_for_category, category_playlists_urls)
        )

        log('{0} raw playlists fetched...'.format(len(playlists)))

        deduped_playlists = dedupe(playlists)

        log('{0} unique playlists fetched...'.format(len(deduped_playlists)))

        self.__db.add_all(deduped_playlists.values())
        self.__db.commit()

        log('Finished fetching public playlists')

    def get_artists_from_playlists(self):
        log('Started fetching objects from public playlists...')

        playlists = self.__db.query(Playlist).all()

        log('{0} playlists to scrape for artists'.format(len(playlists)))

        artists = flatten(self.__parallel_map(get_artists_from_playlist, playlists))

        log('{0} raw artists scraped...'.format(len(artists)))

        deduped_artists = dedupe(artists)

        log('{0} unique artists scraped...'.format(len(deduped_artists)))

        self.__db.add_all(deduped_artists.values())
        self.__db.commit()

        log('Finished fetching artists from playlists')

    def get_related_artists(self):
        artists = self.__db.query(Artist).all()
        original_artists = {artist.id for artist in artists}

        log('Starting to scrape artists related to playlist artists...')

        related_artists = flatten(self.__parallel_map(get_related_artists, artists))

        log('{0} raw artists'.format(len(artists) + len(related_artists)))

        deduped_artists = dedupe(artists + related_artists)

        log('{0} unique artists'.format(len(deduped_artists)))

        log('Finished scraping related artists...')

        self.__db.add_all([artist for id, artist in deduped_artists.items() 
                            if id not in original_artists])
        self.__db.commit()

        log('Finished fetching related artists')

    def get_top_tracks_from_artists(self):
        log('Retrieving objects from db...')
        artists_from_db = {a.id: a for a in self.__db.query(Artist).all()}
        self.__db.query(Artist).delete() # to avoid duplicates existing once top tracks are saved
        self.__db.commit()

        log('Finished retrieving objects from db...')
        log('{0} artists to scrape for top tracks...'.format(len(artists_from_db)))

        top_track_objs = self.__parallel_map(get_top_tracks_from_artist, artists_from_db.values())

        log('Finished fetching objects from Spotify API...')

        raw_tracks, raw_albums, raw_artists = zip(*top_track_objs)
        tracks, albums, artists = flatten(raw_tracks), flatten(raw_albums), flatten(raw_artists)
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

        log('refreshing track relationships')
        final_tracks = refresh_relationships(deduped_tracks.values(), {
            'artists': deduped_artists,
            'album': deduped_albums
        })

        log('refreshing album relationships')
        final_albums = refresh_relationships(deduped_albums.values(), {
            'artists': deduped_artists
        })

        final_artists = deduped_artists.values()

        log('Finished refreshing relationships...')

        self.__db.expunge_all() # to get rid of duplicate objects cached by the db session
        self.__db.add_all(final_tracks)
        self.__db.commit()

        log('Finished committing top tracks and relationships')

    def build_related_artists_graph(self):
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

    def get_genres(self):
        artists_from_db = {a.id: a for a in self.__db.query(Artist).all()}

        log('{0} artists to scrape for genres'.format(len(artists_from_db)))

        artist_genres = dict(self.__parallel_map(get_genres_from_object, artists_from_db.values()))
        log('Finished fetching genres from artists on Spotify')

        genres = {}
        for a_id, genre_set in artist_genres.items():
            for genre in genre_set:
                if genre not in genres:
                    genres[genre] = Genre(name=genre)

                genres[genre].artists.append(artists_from_db[a_id])

        log('Finished deduplicating genres and generating objects')

        self.__db.add_all(genres.values())
        self.__db.commit()

        # albums_from_db = {a.id: a for a in self.__db.query(Album).all()}

        # log('{0} artists to scrape for genres'.format(len(albums_from_db)))

        # album_genres = dict(self.__parallel_map(get_genres_from_object, albums_from_db.values()))
        # log('Finished fetching genres from albums on Spotify')

        # for a_id, genre_set in album_genres.items():
        #     for genre in genre_set:
        #         if genre not in genres:
        #             genres[genre] = Genre(name=genre)

        #         genres[genre].albums.append(albums_from_db[a_id])

    def get_objects_from_playlists(self):
        playlists_from_db = {p.id: p for p in self.__db.query(Playlist).all()}
        tracks_from_db = {t.id: t for t in self.__db.query(Track).all()}
        albums_from_db = {a.id: a for a in self.__db.query(Album).all()}
        artists_from_db = {a.id: a for a in self.__db.query(Artist).all()}

        log('Started fetching objects from public playlists...')
        
        playlist_objs = self.__parallel_map(get_objects_from_playlist, playlists_from_db.values())
        playlist_tracks = {id: {t.id for t in tracks if {a.id for a in t.artists}.issubset(artists_from_db.keys())}
                                for (id, tracks, albums, artists) in playlist_objs}

        log('Finished fetching objects from Spotify API...')

        raw_p, raw_t, raw_alb, raw_art = zip(*playlist_objs)
        tracks, albums = flatten(raw_t), flatten(raw_alb)

        log('raw counts: {0} tracks, {1} albums'.format(
            len(tracks), 
            len(albums)
        ))

        deduped_tracks = dedupe(list(tracks_from_db.values()) + tracks)
        deduped_albums = dedupe(list(albums_from_db.values()) + albums)

        log('Finished deduplicating objects...')
        log('Unique counts: {0} tracks, {1} albums'.format(
            len(deduped_tracks), 
            len(deduped_albums)
        ))

        tracks_to_save = [t for t in deduped_tracks.values() 
                            if t.id not in tracks_from_db
                            and {a.id for a in t.artists}.issubset(artists_from_db.keys())]
        track_relations = {}
        for t in tracks_to_save:
            track_relations[t.id] = {
                'album': t.album.id,
                'artists': {a.id for a in t.artists}
            }
            t.album, t.artists = None, []

        albums_to_save = [a for a in deduped_albums.values() if a.id not in albums_from_db]
        album_artists = {}
        for a in albums_to_save:
            album_artists[a.id] = {artist.id for artist in a.artists}
            a.artists = []

        log('Objects to save: {0} tracks, {1} albums'.format(
            len(tracks_to_save), 
            len(albums_to_save)
        ))

        self.__db.expunge_all() # to get rid of duplicate objects cached by the db session
        self.__db.add_all(tracks_to_save + albums_to_save)
        self.__db.commit()

        log('Finished commiting objects...')
        
        self.__db.expunge_all() # to get rid of duplicate objects cached by the db session

        tracks_from_db = {t.id: t for t in self.__db.query(Track).all()}
        albums_from_db = {a.id: a for a in self.__db.query(Album).all()}
        artists_from_db = {a.id: a for a in self.__db.query(Artist).all()}

        log('Started building relationship graph...')

        track_ids = list(track_relations.keys())
        for i in range(len(track_ids)):
            if i > 0 and i % 1000 == 0:
                log('{0} of {1} track relations created'.format(i, len(track_ids)))

            id = track_ids[i]
            relations = track_relations[id]
            tracks_from_db[id].album = albums_from_db[relations['album']]
            track_artists = [artists_from_db[a_id] for a_id in relations['artists']]
            tracks_from_db[id].artists = track_artists

        album_ids = list(album_artists.keys())
        for i in range(len(album_ids)):
            if i > 0 and i % 1000 == 0:
                log('{0} of {1} album relations created'.format(i, len(album_ids)))

            id = album_ids[i]
            artists_for_album = [artists_from_db[a_id] for a_id in album_artists[id]]
            albums_from_db[id].artists = artists_for_album

        log('Finished building relationship graph...')

        self.__db.commit()

        self.__db.expunge_all()

        playlists_from_db = {p.id: p for p in self.__db.query(Playlist).all()}
        tracks_from_db = {t.id: t for t in self.__db.query(Track).all()}
        
        for playlist_id, track_ids in playlist_tracks.items():
            playlists_from_db[playlist_id].tracks = [tracks_from_db[id] for id in track_ids]

        log('Started commiting playlist relationships...')

        self.__db.commit()

        log('Finished fetching objects from public playlists')

    def download_raw_previews(self):
        tracks_from_db = self.__db.query(Track).filter(Track.preview_url != None).limit(100)
        self.__parallel_map(save_raw_preview, tracks_from_db)
