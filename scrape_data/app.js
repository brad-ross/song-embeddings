const parseArgs = require('minimist')
const connectToDb = require('./db')

const {Artist} = require('./models')

const {scrapePublicPlaylistsForArtists} = require('./spotify_track_scraper')

const args = parseArgs(process.argv.slice(2))
const DONT_SCRAPE_ARTISTS_FLAG = 'no_artists'

function getArtists() {
  return Artist.find({}).exec()
}

connectToDb()
  .then(() => {
    const artistsPromise = args[DONT_SCRAPE_ARTISTS_FLAG] ? 
                            getArtists() : 
                            scrapePublicPlaylistsForArtists()
    
    return artistsPromise.then(tracks => console.log(tracks.length))
  })
  .catch(err => console.log(err))