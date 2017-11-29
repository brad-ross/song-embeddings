from scraper import Scraper
from time import time
from .utils import log

genres_to_scrape = ['classical', 'rap', 'hip hop', 'edm', 'pop', 'house', 'indie rock', 
                    'latin', 'rock', 'r&b', 'metal', 'funk', 'folk', 'country', 'jazz']

if __name__ == '__main__':
    print(__package__)
    scraper = Scraper()
    # scraper.get_public_playlists()
    # scraper.get_artists_from_playlists()
    # scraper.get_related_artists()
    # scraper.get_top_tracks_from_artists()
    # scraper.build_related_artists_graph()
    # scraper.get_genres()
    
    for genre in genres_to_scrape:
        log('Now scraping 1000 songs for genre: {}'.format(genre))
        scraper.download_raw_previews_for_genre(genre)
