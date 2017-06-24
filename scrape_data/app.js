const express = require('express')
const auth = require('./auth')

const app = express()

app.get('/', (req, res) => {
  res.status(200).send('This is the future home of my Spotify scraper!').end()
})

// Start the server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`)
  console.log('Press Ctrl+C to quit.')
  auth.getClientCredAccessToken()
    .then(token => console.log(token))
})