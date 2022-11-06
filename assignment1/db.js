//pickup line generation app 

var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var pickupSchema = new Schema({
    pickup: String
});

var Pickup = mongoose.model('Pickup', pickupSchema);

module.exports = Pickup;

// Path: index.js

var express = require('express');

var app = express();

var mongoose = require('mongoose');
mongoose.connect('mongodb://localhost/pickup');

var Pickup = require('./models/pickup');

app.get('/', function(req, res) {
    Pickup.find(function(err, pickup) {
        if (err) {
            res.send(err);
        }
        res.json(pickup);
    });
});

app.listen(3000, function() {
    console.log('Listening on port 3000');
});

//genererate new  pickup line based on data found in db
function generatePickupLine() {
    var randomIndex = Math.floor(Math.random() * pickupLines.length);
    return pickupLines[randomIndex];
}
