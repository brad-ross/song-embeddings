const getConfig = require('./config')
const fetch = require('node-fetch')
const URLSearchParams = require('url-search-params')
const querystring = require('querystring')

const MAX_RETRIES = 3
const RETRY_DELAY = 1500
const UNAUTHORIZED_CODE = 401

// Using global state like this, especially in a concurrent environment, probably isn't the best
// idea, but it's easy enough
let currAccessToken = null

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

function refreshAccessToken() {
  console.log(`Old Access Token: ${currAccessToken}`)
  return getClientCredAccessToken()
    .then(newToken => {
      currAccessToken = newToken
      console.log(`New Access Token: ${currAccessToken}`)
    })
}

function spotifyRequest(url, options={}, numRetries=0) {
  const qs = !options.qs ? '' : '?' + querystring.stringify(options.qs)

  if (numRetries > MAX_RETRIES) {
    // TODO: better handling of max retries
    console.log(`MAX_RETRIES exceeded for ${url + qs}`)
    return Promise.resolve()
  }
  
  if (!options.headers) options.headers = {}
  options.headers.Authorization = 'Bearer ' + currAccessToken

  return fetch(url + qs, options)
    .then(res => {
      return res.json()
        .then(body => ({raw: res, body: body}))
    })
    .then(resObj => {
      if (!resObj.raw.ok) {
        console.log(resObj.body)
        // This block handles the case where the provided auth token expired.
        // In this case, a new auth token must be retrieved, and then the
        // request can be retried
        if (resObj.body.error && resObj.body.error.status === UNAUTHORIZED_CODE) {
          return refreshAccessToken()
            .then(() => spotifyRequest(url, options, numRetries))
        }

        console.log(res.raw.statusText, url + qs)
      }

      return resObj.body
    })
    .catch(err => {
      console.log(`Error while making request with URL ${url + qs}`)
      console.log(err)
      console.log('retrying...')
      return new Promise((resolve, reject) => 
        setTimeout(() => 
          spotifyRequest(url, options, numRetries + 1)
            .then(res => resolve(res)),
      RETRY_DELAY))
    })
}

function makeRateLimitedRequests(items, fn, delay, argsForFn=[]) {
  return Promise.all(items.map((item, i) =>
    new Promise((resolve, reject) => 
      setTimeout(
        () => fn(item, i, resolve, reject, ...argsForFn), 
      i*delay)
    )
  ))
}

module.exports = {getClientCredAccessToken, spotifyRequest, makeRateLimitedRequests}