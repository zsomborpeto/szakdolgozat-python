from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify
import service.traffic_service
from dateutil.parser import parse
import pytz
import math
import datetime

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

traffic_service = service.traffic_service.TrafficService()


@app.route('/traffic/daily', methods=['POST'])
def home():
    start = request.json.get('start')
    end = request.json.get('end')
    return traffic_service.get_daily_traffic_between(start, end).to_json(orient="split", index=False, force_ascii=False)


@app.route('/traffic/actual', methods=['POST'])
def ten_min():
    time = request.json.get('time')
    temp = request.json.get('temp')
    type = request.json.get('type')
    date = parse(time).astimezone(pytz.timezone("Europe/Budapest"))

    service.traffic_service.TrafficService.prev_szeged = 0
    service.traffic_service.TrafficService.prev_ujszeged = 0

    return jsonify({"toSzeged": str(traffic_service.get_10min_predictions("szeged", date, temp, type)),
                    "toUjszeged": str(traffic_service.get_10min_predictions("ujszeged", date, temp, type))})


@app.route('/traffic/ten-min', methods=['POST'])
def next_hour():
    time = request.json.get('time')
    temp = request.json.get('temp')
    direction = request.json.get('direction')
    date = parse(time).astimezone(pytz.timezone("Europe/Budapest"))
    date = date + datetime.timedelta(minutes=-10)
    date = date.replace(minute=__get_minute(date.minute), second=0, microsecond=0)

    service.traffic_service.TrafficService.prev_szeged = 0
    service.traffic_service.TrafficService.prev_ujszeged = 0

    car_return_data = []
    bicycle_return_data = []
    ped_return_data = []

    for x in range(26):
        new_date = date + datetime.timedelta(minutes=x * 10)

        car_pred = traffic_service.get_10min_predictions(direction, new_date, temp, "car")
        if x != 0:
            car_return_data.append({"value": str(car_pred), "date": new_date})

        bicycle_pred = traffic_service.get_10min_predictions(direction, new_date, temp, "bicycle")
        if x != 0:
            bicycle_return_data.append({"value": str(bicycle_pred), "date": new_date})

        pedestrian_pred = traffic_service.get_10min_predictions(direction, new_date, temp, "pedestrian")
        if x != 0:
            ped_return_data.append({"value": str(pedestrian_pred), "date": new_date})

    return jsonify({"car": car_return_data, "bicycle": bicycle_return_data, "pedestrian": ped_return_data})


@app.route('/traffic/hour', methods=['POST'])
def next_day():
    time = request.json.get('time')
    temp = request.json.get('temp')
    direction = request.json.get('direction')
    date = parse(time).astimezone(pytz.timezone("Europe/Budapest"))
    date = date.replace(minute=0, second=0, microsecond=0)

    car_return_data = []
    bicycle_return_data = []
    ped_return_data = []

    for x in range(25):
        new_date = date + datetime.timedelta(hours=x)
        car_return_data.append({"value": str(traffic_service.get_hour_predictions(direction, new_date, temp, "car")),
                                "date": new_date})
        bicycle_return_data.append({"value": str(traffic_service.get_hour_predictions(direction, new_date, temp, "bicycle")),
                                    "date": new_date})
        ped_return_data.append({"value": str(traffic_service.get_hour_predictions(direction, new_date, temp, "pedestrian")),
                                "date": new_date})

    return jsonify({"car": car_return_data, "bicycle": bicycle_return_data, "pedestrian": ped_return_data})


@app.route('/traffic/day', methods=['POST'])
def next_week():
    time = request.json.get('time')
    temp = request.json.get('temp')
    direction = request.json.get('direction')
    date = parse(time).astimezone(pytz.timezone("Europe/Budapest"))
    date = date.replace(hour=0, minute=0, second=0, microsecond=0)

    car_return_data = []
    bicycle_return_data = []
    ped_return_data = []

    for x in range(8):
        new_date = date + datetime.timedelta(days=x)
        car_return_data.append({"value": str(traffic_service.get_day_predictions(direction, new_date, temp, "car")),
                                "date": new_date})
        bicycle_return_data.append({"value": str(traffic_service.get_day_predictions(direction, new_date, temp, "bicycle")),
                                    "date": new_date})
        ped_return_data.append({"value": str(traffic_service.get_day_predictions(direction, new_date, temp, "pedestrian")),
                                "date": new_date})

    return jsonify({"car": car_return_data, "bicycle": bicycle_return_data, "pedestrian": ped_return_data})


def __get_minute(minute):
    return math.floor(minute / 10) * 10


app.run(host='192.168.0.22')
