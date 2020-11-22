import pandas as pd
import glob

traffic_path = '../../SmartCity-master/traffic\\'
daily_folder = "daily"
separator = ';'
date_column = 'Dátum'


def read_all_traffic_daily():
    path = traffic_path + "AP1\\" + daily_folder
    all_files = glob.glob(path + "/*.csv")

    csvs = []

    for file in all_files:
        csvs.append(pd.read_csv(file, sep=separator, index_col=None, header=0))

    return convert_to_date(pd.concat(csvs, axis=0, ignore_index=True))


def convert_to_date(data):
    data[date_column] = data[date_column].astype('datetime64[ns]').dt.date
    return data
