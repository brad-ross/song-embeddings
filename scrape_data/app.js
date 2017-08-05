const parseArgs = require('minimist')
const connectToDb = require('./db')

const {Artist, Track} = require('./models')

const {scrapePublicPlaylistsForArtists, 
       scrapeTopTracksFromArtists} = require('./spotify_track_scraper')

const args = parseArgs(process.argv.slice(2))
const DONT_SCRAPE_ARTISTS_FLAG = 'no_artists'
const DONT_SCRAPE_SONGS_FLAG = 'no_songs'

function getArtists() {
  return Artist.find({}).exec()
}

function getTracks() {
  return Track.find({}).exec()
}

connectToDb()
  .then(() => args[DONT_SCRAPE_ARTISTS_FLAG] ? 
                            getArtists() : 
                            scrapePublicPlaylistsForArtists())
  .then(artists => args[DONT_SCRAPE_SONGS_FLAG] ?
                            getTracks() :
                            scrapeTopTracksFromArtists(artists))
  .then((promises) => {
    const [tracks, artists] = promises
    console.log(tracks.length)
  })
  .catch(err => console.log(err))