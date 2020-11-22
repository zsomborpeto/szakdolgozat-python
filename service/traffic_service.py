import math
import joblib
import pandas as pd
import repository.csv_reader
import datetime as dt
from tensorflow.keras.models import load_model


class TrafficService:
    date_column = 'Dátum'

    fix_holidays = [
        "1-1",
        "3-15",
        "5-1",
        "8-20",
        "10-23",
        "11-1",
        "12-25",
        "12-26",
    ]

    moving_holidays = [
        "2021-4-2",
        "2021-4-4",
        "2021-4-5",
        "2021-5-23",
        "2021-5-24",
        "2022-4-15",
        "2022-4-17",
        "2022-4-18",
        "2022-6-5",
        "2022-6-6"
    ]

    car_model_belvaros = load_model('../model/10min_Belvaros_fele_autok.h5')
    car_model_ujszeged = load_model('../model/10min_Ujszeged_fele_autok.h5')
    car_scaler_belvaros = joblib.load('../model/10min_Belvaros_fele_autok.pkl')
    car_scaler_ujszeged = joblib.load('../model/10min_Ujszeged_fele_autok.pkl')

    car_prev_model_belvaros = load_model('../model/10min_prev_Szeged_fele_autok.h5')
    car_prev_scaler_belvaros = joblib.load('../model/10min_prev_Szeged_fele_autok.pkl')
    car_prev_model_ujszeged = load_model('../model/10min_prev_Újszeged_fele_autok.h5')
    car_prev_scaler_ujszeged = joblib.load('../model/10min_prev_Újszeged_fele_autok.pkl')

    ped_model_belvaros = load_model('../model/10min_Belvaros_fele_gyalogosok.h5')
    ped_model_ujszeged = load_model('../model/10min_Ujszeged_fele_gyalogosok.h5')
    ped_scaler_belvaros = joblib.load('../model/10min_Belvaros_fele_gyalogosok.pkl')
    ped_scaler_ujszeged = joblib.load('../model/10min_Ujszeged_fele_gyalogosok.pkl')

    bike_model_belvaros = load_model('../model/10min_Belvaros_fele_biciklisek.h5')
    bike_model_ujszeged = load_model('../model/10min_Ujszeged_fele_biciklisek.h5')
    bike_scaler_belvaros = joblib.load('../model/10min_Belvaros_fele_biciklisek.pkl')
    bike_scaler_ujszeged = joblib.load('../model/10min_Ujszeged_fele_biciklisek.pkl')

    hour_car_model_belvaros = load_model('../model/hour_Szeged_fele_autok.h5')
    hour_car_model_ujszeged = load_model('../model/hour_Újszeged_fele_autok.h5')
    hour_car_scaler_belvaros = joblib.load('../model/hour_Szeged_fele_autok.pkl')
    hour_car_scaler_ujszeged = joblib.load('../model/hour_Újszeged_fele_autok.pkl')

    hour_ped_model_belvaros = load_model('../model/hour_Szeged_fele_gyalogosok.h5')
    hour_ped_model_ujszeged = load_model('../model/hour_Újszeged_fele_gyalogosok.h5')
    hour_ped_scaler_belvaros = joblib.load('../model/hour_Szeged_fele_gyalogosok.pkl')
    hour_ped_scaler_ujszeged = joblib.load('../model/hour_Újszeged_fele_gyalogosok.pkl')

    hour_bike_model_belvaros = load_model('../model/hour_Szeged_fele_biciklisek.h5')
    hour_bike_model_ujszeged = load_model('../model/hour_Újszeged_fele_biciklisek.h5')
    hour_bike_scaler_belvaros = joblib.load('../model/hour_Szeged_fele_biciklisek.pkl')
    hour_bike_scaler_ujszeged = joblib.load('../model/hour_Újszeged_fele_biciklisek.pkl')

    day_car_model_belvaros = load_model('../model/day_Szeged_fele_autok.h5')
    day_car_model_ujszeged = load_model('../model/day_Újszeged_fele_autok.h5')
    day_car_scaler_belvaros = joblib.load('../model/day_Szeged_fele_autok.pkl')
    day_car_scaler_ujszeged = joblib.load('../model/day_Újszeged_fele_autok.pkl')

    day_ped_model_belvaros = load_model('../model/day_Szeged_fele_gyalogosok.h5')
    day_ped_model_ujszeged = load_model('../model/day_Újszeged_fele_gyalogosok.h5')
    day_ped_scaler_belvaros = joblib.load('../model/day_Szeged_fele_gyalogosok.pkl')
    day_ped_scaler_ujszeged = joblib.load('../model/day_Újszeged_fele_gyalogosok.pkl')

    day_bike_model_belvaros = load_model('../model/day_Szeged_fele_biciklisek.h5')
    day_bike_model_ujszeged = load_model('../model/day_Újszeged_fele_biciklisek.h5')
    day_bike_scaler_belvaros = joblib.load('../model/day_Szeged_fele_biciklisek.pkl')
    day_bike_scaler_ujszeged = joblib.load('../model/day_Újszeged_fele_biciklisek.pkl')

    prev_szeged = 0
    prev_ujszeged = 0

    def get_daily_traffic_between(self, start, end):
        df = repository.csv_reader.read_all_traffic_daily()
        return self.__add_weekend_column(self.__format_date(self.__filter_between_dates(start, end, df)))

    def get_10min_predictions(self, direction, date, temp, type):
        global model, scaler
        if direction == "szeged":
            if type == "car":
                model = self.car_model_belvaros
                scaler = self.car_scaler_belvaros
            elif type == "pedestrian":
                model = self.ped_model_belvaros
                scaler = self.ped_scaler_belvaros
            elif type == "bicycle":
                model = self.bike_model_belvaros
                scaler = self.bike_scaler_belvaros
        else:
            if type == "car":
                model = self.car_model_ujszeged
                scaler = self.car_scaler_ujszeged
            elif type == "pedestrian":
                model = self.ped_model_ujszeged
                scaler = self.ped_scaler_ujszeged
            elif type == "bicycle":
                model = self.bike_model_ujszeged
                scaler = self.bike_scaler_ujszeged

        if type == "car":
            if self.prev_szeged != 0 and direction == "szeged":
                model = self.car_prev_model_belvaros
                scaler = self.car_prev_scaler_belvaros
                predict_data = [[self.prev_szeged, temp, date.hour, self.__get_minute(date), date.weekday() == 0,
                                 date.weekday() == 1, date.weekday() == 2, date.weekday() == 3, date.weekday() == 4,
                                 date.weekday() == 5, date.weekday() == 6, self.__is_curfew(date), False,
                                 self.__isEducation(date), self.__is_weekend(date), self.__is_holiday(date)]]
                self.prev_szeged = math.floor(model.predict(scaler.transform(predict_data))[0][0])
                return self.prev_szeged

            elif self.prev_ujszeged != 0 and direction == "ujszeged":
                model = self.car_prev_model_ujszeged
                scaler = self.car_prev_scaler_ujszeged
                predict_data = [[self.prev_ujszeged, temp, date.hour, self.__get_minute(date), date.weekday() == 0,
                                 date.weekday() == 1, date.weekday() == 2, date.weekday() == 3, date.weekday() == 4,
                                 date.weekday() == 5, date.weekday() == 6, self.__is_curfew(date), False,
                                 self.__isEducation(date), self.__is_weekend(date), self.__is_holiday(date)]]
                self.prev_ujszeged = math.floor(model.predict(scaler.transform(predict_data))[0][0])
                return self.prev_ujszeged

        predict_data = [[temp, date.hour, self.__get_minute(date), date.weekday() == 0, date.weekday() == 1,
                         date.weekday() == 2, date.weekday() == 3, date.weekday() == 4, date.weekday() == 5,
                         date.weekday() == 6, self.__is_curfew(date), False, self.__isEducation(date),
                         self.__is_weekend(date), self.__is_holiday(date)]]

        if type == "car":
            if self.prev_szeged == 0 and direction == "szeged":
                self.prev_szeged = math.floor(model.predict(scaler.transform(predict_data))[0][0])
                return self.prev_szeged
            elif self.prev_ujszeged == 0 and direction == "ujszeged":
                self.prev_ujszeged = math.floor(model.predict(scaler.transform(predict_data))[0][0])
                return self.prev_ujszeged

        return math.floor(model.predict(scaler.transform(predict_data))[0][0])

    def get_hour_predictions(self, direction, date, temp, type):
        if direction == "szeged":
            if type == "car":
                model = self.hour_car_model_belvaros
                scaler = self.hour_car_scaler_belvaros
            elif type == "pedestrian":
                model = self.hour_ped_model_belvaros
                scaler = self.hour_ped_scaler_belvaros
            elif type == "bicycle":
                model = self.hour_bike_model_belvaros
                scaler = self.hour_bike_scaler_belvaros
        else:
            if type == "car":
                model = self.hour_car_model_ujszeged
                scaler = self.hour_car_scaler_ujszeged
            elif type == "pedestrian":
                model = self.hour_ped_model_ujszeged
                scaler = self.hour_ped_scaler_ujszeged
            elif type == "bicycle":
                model = self.hour_bike_model_ujszeged
                scaler = self.hour_bike_scaler_ujszeged

        predict_data = [[temp, date.hour, date.weekday() == 0, date.weekday() == 1, date.weekday() == 2,
                         date.weekday() == 3, date.weekday() == 4, date.weekday() == 5, date.weekday() == 6,
                         self.__is_curfew(date), False, self.__isEducation(date), self.__is_weekend(date),
                         self.__is_holiday(date)]]

        return math.floor(model.predict(scaler.transform(predict_data))[0][0])

    def get_day_predictions(self, direction, date, temp, type):
        if direction == "szeged":
            if type == "car":
                model = self.day_car_model_belvaros
                scaler = self.day_car_scaler_belvaros
            elif type == "pedestrian":
                model = self.day_ped_model_belvaros
                scaler = self.day_ped_scaler_belvaros
            elif type == "bicycle":
                model = self.day_bike_model_belvaros
                scaler = self.day_bike_scaler_belvaros
        else:
            if type == "car":
                model = self.day_car_model_ujszeged
                scaler = self.day_car_scaler_ujszeged
            elif type == "pedestrian":
                model = self.day_ped_model_ujszeged
                scaler = self.day_ped_scaler_ujszeged
            elif type == "bicycle":
                model = self.day_bike_model_ujszeged
                scaler = self.day_bike_scaler_ujszeged

        predict_data = [[temp, date.weekday() == 0, date.weekday() == 1, date.weekday() == 2,
                         date.weekday() == 3, date.weekday() == 4, date.weekday() == 5, date.weekday() == 6,
                         False, False, self.__isEducation(date), self.__is_weekend(date),
                         self.__is_holiday(date)]]

        return math.floor(model.predict(scaler.transform(predict_data))[0][0])

    def __is_curfew(self, date):
        return date.hour >= 20 or (0 <= date.hour < 5)

    def __is_weekend(self, date):
        return date.weekday() >= 5

    def __is_holiday(self, date):
        return self.moving_holidays.__contains__(str(date.date())) or self.fix_holidays.__contains__(str(date.month) + "-" + str(date.day))

    def __isEducation(self, date):
        return not self.__is_holiday(date) and not self.__is_weekend(date)

    def __get_minute(self, date):
        return math.floor(date.minute / 10) * 10

    def __filter_between_dates(self, start, end, df):
        return df[(df[self.date_column] >= pd.to_datetime(start).normalize().tz_localize(None))
                  & (df[self.date_column] <= pd.to_datetime(end).normalize().tz_localize(None))]

    def __format_date(self, df):
        df[self.date_column] = df[self.date_column].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
        return df

    def __add_weekend_column(self, df):
        df['Weekend'] = pd.DatetimeIndex(df[self.date_column]).dayofweek >= 5
        return df



