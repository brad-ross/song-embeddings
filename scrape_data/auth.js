const getConfig = require('./config')
const fetch = require('node-fetch')
const URLSearchParams = require('url-search-params')

function getClientCredAccessToken() {
  const {client_id, client_secret} = getConfig().spotify_auth
  const params = new URLSearchParams()
  params.append('grant_type', 'client_credentials')

  return fetch('https://accounts.spotify.com/api/token', {
      method: 'POST',
      headers: {
        'Content-Type' : 'application/x-www-form-urlencoded',
        'Authorization': 'Basic ' + (new Buffer(client_id + ':' + client_secret).toString('base64'))
      },
      body: params
    })
    .then(res => res.json())
    .then(json => json.access_token)
}

module.exports = {getClientCredAccessToken}