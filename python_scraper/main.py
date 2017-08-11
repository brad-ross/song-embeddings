from spotify import Client
from scraper import Scraper
from time import time

if __name__ == '__main__':
    scraper = Scraper()
    #scraper.get_public_playlists()
    #scraper.get_objects_from_playlists()
    scraper.build_related_artists_graph()