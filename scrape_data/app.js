const express = require('express')
const connectToDb = require('./db')
const auth = require('./auth')

connectToDb()
  .then(() => auth.getClientCredAccessToken())
  .then(token => {
    console.log(token)
  })