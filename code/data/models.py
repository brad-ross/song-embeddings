from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship

Base = declarative_base()

track_artists = Table('track_artists', Base.metadata,
    Column('track_id', ForeignKey('tracks.id'), primary_key=True),
    Column('artist_id', ForeignKey('artists.id'), primary_key=True)
)

playlist_tracks = Table('playlist_tracks', Base.metadata,
    Column('playlist_id', ForeignKey('playlists.id'), primary_key=True),
    Column('track_id', ForeignKey('tracks.id'), primary_key=True)
)

class Track(Base):
    __tablename__ = 'tracks'

    id = Column(String, primary_key=True)
    name = Column(String)
    preview_url = Column(String)
    album_id = Column(String, ForeignKey('albums.id'))
    is_top_track = Column(Boolean, default=False)

    artists = relationship('Artist', secondary=track_artists, back_populates='tracks')
    album = relationship('Album', back_populates='tracks')
    playlists = relationship('Playlist', secondary=playlist_tracks, back_populates='tracks')

    @staticmethod
    def init_from_api(api_obj):
        return Track(id=api_obj['id'], 
                     name=api_obj['name'], 
                     preview_url=api_obj['preview_url'])

class Playlist(Base):
    __tablename__ = 'playlists'

    id = Column(String, primary_key=True)
    name = Column(String)
    user_id = Column(String)
    snapshot_id = Column(String)

    tracks = relationship('Track', secondary=playlist_tracks, back_populates='playlists')

    @staticmethod
    def init_from_api(api_obj):
        return Playlist(id=api_obj['id'], 
                        name=api_obj['name'], 
                        user_id=api_obj['owner']['id'], 
                        snapshot_id=api_obj['snapshot_id'])

album_artists = Table('album_artists', Base.metadata,
    Column('album_id', ForeignKey('albums.id'), primary_key=True),
    Column('artist_id', ForeignKey('artists.id'), primary_key=True)
)

album_genres = Table('album_genres', Base.metadata,
    Column('album_id', ForeignKey('albums.id'), primary_key=True),
    Column('genre', ForeignKey('genres.name'), primary_key=True)
)

class Album(Base):
    __tablename__ = 'albums'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)

    tracks = relationship('Track', back_populates='album')
    artists = relationship('Artist', secondary=album_artists, back_populates='albums')
    genres = relationship('Genre', secondary=album_genres, back_populates='albums')

    @staticmethod
    def init_from_api(api_obj):
        return Album(id=api_obj['id'], name=api_obj['name'])

artist_genres = Table('artist_genres', Base.metadata,
    Column('artist_id', ForeignKey('artists.id'), primary_key=True),
    Column('genre', ForeignKey('genres.name'), primary_key=True)
)

related_artists = Table('related_artists', Base.metadata,
    Column('artist_id', ForeignKey('artists.id'), primary_key=True),
    Column('related_artist_id', ForeignKey('artists.id'), primary_key=True)
)

class Artist(Base):
    __tablename__ = 'artists'

    id = Column(String, primary_key=True)
    name = Column(String)

    tracks = relationship('Track', secondary=track_artists, back_populates='artists')
    albums = relationship('Album', secondary=album_artists, back_populates='artists')
    genres = relationship('Genre', secondary=artist_genres, back_populates='artists')
    related_artists = relationship('Artist', 
                                   secondary=related_artists, 
                                   back_populates='related_artists',
                                   primaryjoin=id==related_artists.c.artist_id,
                                   secondaryjoin=id==related_artists.c.related_artist_id)

    @staticmethod
    def init_from_api(api_obj):
        return Artist(id=api_obj['id'], name=api_obj['name'])

class Genre(Base):
    __tablename__ = 'genres'

    name = Column(String, primary_key=True)

    albums = relationship('Album', secondary=album_genres, back_populates='genres')
    artists = relationship('Artist', secondary=artist_genres, back_populates='genres')
