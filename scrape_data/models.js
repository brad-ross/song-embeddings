const mongoose = require('mongoose')
const Schema = mongoose.Schema

const trackSchema = new Schema({
  _id: Schema.Types.ObjectId,
  name: String,
  preview_url: String
})
const Track = mongoose.model('Track', trackSchema)

module.exports = {Track}