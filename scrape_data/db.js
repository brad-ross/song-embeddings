const getConfig = require('./config')
const mongoose = require('mongoose')
mongoose.Promise = Promise

function connectToDb() {
  const {db_url} = getConfig().db_auth
  return mongoose.connect(db_url)
}

module.exports = connectToDb