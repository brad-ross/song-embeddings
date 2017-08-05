const _ = require('lodash')
const getConfig = require('./config')
const {spotifyRequest, makeRateLimitedRequests} = require('./spotify_client')
const {Track, Artist} = require('./models')

const SPOTIFY_BASE_URL = 'https://api.spotify.com/v1'
const NUM_CATEGORIES = 50
const CATS_TO_EXCLUDE = new Set(getConfig().scraping.categories_to_exclude)
const REQUEST_DELAY = 500
const PRINT_MULTIPLE = 100
const DEFAULT_COUNTRY = 'US'

function scrapePublicPlaylistsForArtists() {
  return getPlaylistTrackUrlsFromAllCategs()
    .then(playlistUrls => getTracksFromPlaylistUrls(playlistUrls))
    .then(tracks => getArtistsFromTracks(tracks))
    .then(artists => {
      return getRelatedArtistsFromArtists(artists)
        .then(relatedArtists => artists.concat(relatedArtists))
    })
    .then(artists => saveArtists(_.uniqBy(artists, '_id')))
}

function getPlaylistTrackUrlsFromAllCategs() {
  const options = {qs: {limit: NUM_CATEGORIES}}
  return spotifyRequest(SPOTIFY_BASE_URL + '/browse/categories', options)
    .then(res => res.categories.items.filter(category => !CATS_TO_EXCLUDE.has(category.name)))
    .then(categories => categories.map(category => category.href + '/playlists'))
    .then(categoryUrls => Promise.all(categoryUrls.map(url => spotifyRequest(url))))
    .then(categoryPlaylistSets => 
      categoryPlaylistSets.map(categoryPlaylists => categoryPlaylists.playlists.items)
    )
    .then(playlistSets => 
      _.flatMap(playlistSets, playlists => playlists.map(playlist => playlist.tracks.href))
    )
}

function getTracksFromPlaylistUrls(playlistUrls) {
  return makeRateLimitedRequests(playlistUrls, 
    (url, i, resolve, reject) => getTracksFromPlaylistUrl(url)
      .then(tracks => {
        if (i % PRINT_MULTIPLE == 0) 
          console.log(`${i} out of ${playlistUrls.length} playlists scraped`)
        resolve(tracks)
      })
      .catch(err => reject(err)), 
  REQUEST_DELAY)
    .then(tracks => _.uniqBy(_.flatten(tracks), 'id'))
}

function getTracksFromPlaylistUrl(playlistUrl) {
  return spotifyRequest(playlistUrl)
    .then(res => res.items.map(item => {
      if (!item.track) return null
      return item.track
    }))
    .then(tracks => tracks.filter(track => track))
}

function getArtistsFromTracks(tracks) {
  return makeRateLimitedRequests(tracks, 
    (track, i, resolve, reject) => getArtistsFromTrack(track)
      .then(artists => {
        if (i % PRINT_MULTIPLE == 0) 
          console.log(`${i} out of ${tracks.length} tracks scraped for artists`)
        resolve(artists)
      })
      .catch(err => reject(err)),
  REQUEST_DELAY)
    .then(artists => _.uniqBy(_.flatten(artists), '_id'))
}

function getArtistsFromTrack(rawTrack) {
  return Promise.all(rawTrack.artists.map(artist => spotifyRequest(artist.href)))
    .then(artists => artists.map(artist => {
      if (!artist) return null
      return Artist.createFromSpotify(artist)
    }))
    .then(artists => artists.filter(artist => artist))
}

function getRelatedArtistsFromArtists(artists) {
  return makeRateLimitedRequests(artists, 
    (artist, i, resolve, reject) => getRelatedArtistsFromArtist(artist)
      .then(relatedArtists => {
        if (i % PRINT_MULTIPLE == 0)
          console.log(`${i} out of ${artists.length} artists scraped for related artists`)
        resolve(relatedArtists)
      })
      .catch(err => reject(err)), 
  REQUEST_DELAY)
    .then(artists => _.uniqBy(_.flatten(artists), '_id'))
}

function getRelatedArtistsFromArtist(artist) {
  return spotifyRequest(artist.self_url + '/related-artists')
    .then(res => res.artists.map(artist => {
      if (!artist) return null
      return Artist.createFromSpotify(artist)
    }))
    .then(artists => artists.filter(artist => artist))
}

function saveArtists(artists) {
  return Artist.insertMany(artists)
}

function scrapeTopTracksFromArtists(artists) {
  return makeRateLimitedRequests(artists,
    (artist, i, resolve, reject) => getTopTracksForArtist(artist)
      .then(tracks => {
        if (i % PRINT_MULTIPLE == 0)
          console.log(`${i} out of ${artists.length} artists scraped for top tracks`)
        resolve(tracks)
      })
      .catch(err => reject(err)),
  REQUEST_DELAY)
    .then(trackSets => _.uniqBy(_.flatten(trackSets), '_id'))
    .then(tracks => Promise.all([saveTracks(tracks), updateArtists(artists)]))
}

function getTopTracksForArtist(artist) {
  return spotifyRequest(SPOTIFY_BASE_URL + '/artists/' + artist._id + '/top-tracks', 
                        {qs: {country: DEFAULT_COUNTRY}})
    .then(res => res.tracks.map(
      track => {
        artist.tracks.push(track.id)
        return Track.createFromSpotify(track)
      }
    ))
}

function saveTracks(tracks) {
  return Track.insertMany(tracks)
}

function updateArtists(artists) {
  return Artist.updateMany(artists, {new: true})
}

module.exports = {
  scrapePublicPlaylistsForArtists,
  scrapeTopTracksFromArtists
}