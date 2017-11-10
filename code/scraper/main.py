from scraper import Scraper
from time import time

if __name__ == '__main__':
    print(__package__)
    scraper = Scraper()
    # scraper.get_public_playlists()
    # scraper.get_artists_from_playlists()
    # scraper.get_related_artists()
    # scraper.get_top_tracks_from_artists()
    # scraper.build_related_artists_graph()
    # scraper.get_genres()
    
    scraper.download_raw_previews_for_genre('rock')
    scraper.download_raw_previews_for_genre('edm')
