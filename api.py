from flask import Flask, redirect, url_for, request, send_file, jsonify, session
from flask_cors import CORS, cross_origin
import json
import data_interface
import server_services as ss


app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)
app.secret_key = 'verySecretKey'
DI = data_interface.DataInterface()

@app.route('/', methods=['GET'])
def home():
    return 'hello from api'

@app.route('/races', methods=['GET'])
def get_past_races():
    return DI.getPastRaces()

@app.route('/future-races', methods=['GET'])
def get_future_races():
    return DI.getFutureRaces()

@app.route('/current-drivers', methods=['GET'])
def get_current_races():
    return DI.getCurrentDrivers()

@app.route('/driver-details', methods=['POST'])
@cross_origin()
def get_driver_details():
    return DI.getDriverDetails(int(request.json['driverId']))

@app.route('/race-prediction', methods=['POST'])
@cross_origin()
def get_race_prediction():
    ticket = ss.getTicket()
    return DI.getRacePrediction(ticket, int(request.json['raceId']), request.json['weather'])

@app.route('/race-results', methods=['POST'])
@cross_origin()
def get_race_results():
    return DI.getRaceResults(int(request.json['raceId']))

@app.route('/race-details', methods=['POST'])
@cross_origin()
def get_race_details():
    return DI.getRaceDetails(int(request.json['raceId']))

@app.route('/prediction-plot', methods=['POST'])
@cross_origin()
def get_pred_plot():
    return send_file('cache/' + request.json['kind'] + '_' + request.json['ticket'] + '.png', mimetype='image/gif')

@app.route('/race-plot', methods=['POST'])
@cross_origin()
def get_race_plot():
    ticket = ss.getTicket()
    print('Request from POST:', request.json['raceId'])
    DI.getRacePlot(ticket, int(request.json['raceId']))
    return send_file('cache/' + ticket + '.png', mimetype='image/gif')

@app.route('/driver-plot', methods=['POST'])
@cross_origin()
def get_driver_plot():
    ticket = ss.getTicket()
    DI.getDriverCluster(ticket, int(request.json['driverId']))
    return send_file('cache/' + ticket + '.png', mimetype='image/gif')
    
app.run(host='0.0.0.0', port=8080)
