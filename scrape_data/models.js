const mongoose = require('mongoose')
const Schema = mongoose.Schema

const trackSchema = new Schema({
  _id: String,
  name: String,
  artists: [{type: String, ref: 'Artist'}],
  album_name: String,
  self_url: String,
  album_url: String,
  preview_url: String,
  spectrogram_path: String
})
trackSchema.statics.createFromSpotify = (rawTrack) => ({
  _id: rawTrack.id, 
  name: rawTrack.name, 
  artists: rawTrack.artists.map((artist) => artist.id),
  album_name: rawTrack.album.name,
  album_url: rawTrack.album.href,
  preview_url: rawTrack.preview_url,
  spectrogram_path: null
})
const Track = mongoose.model('Track', trackSchema)

const artistSchema = new Schema({
  _id: String,
  name: String,
  genres: [String],
  self_url: String,
  tracks: [{type: String, ref: 'Track'}]
})
artistSchema.statics.createFromSpotify = (rawArtist) => ({
  _id: rawArtist.id,
  name: rawArtist.name,
  genres: rawArtist.genres,
  self_url: rawArtist.href,
  tracks: []
})
const Artist = mongoose.model('Artist', artistSchema)

module.exports = {Track, Artist}