import pandas as pd
import matplotlib.pyplot as plt

file_names = [
    "Aug",
    "Dec",
    "Jun",
    "Nov",
    "Oct",
    "Sep"
]

#format="%d/%b/%Y:%H:%M:%S"

def get_series(logs_df):
  print(logs_df.head())
  date_time_index = pd.to_datetime(logs_df["date_time"])
  series = pd.Series(data=logs_df["size"].values, name='size', index=date_time_index)
  series_grouped_by_hour = series.resample("min").max()
  series_grouped_by_hour.columns=['minute', 'size']
  return series_grouped_by_hour


def get_df(logs_df):
  date_time_column = pd.to_datetime(logs_df["date"].astype(str) + " " + logs_df["time"])
  df = pd.DataFrame({"date_time":date_time_column, "response_code":logs_df["response_code"], "size":logs_df["size"]})
  df.set_index('date_time', inplace=True)
  grouped_by_hours = df.resample('H')["size"].sum()
  return grouped_by_hours


def main():
  for name in file_names:
    logs_df = pd.read_csv(name + "_second_sum.csv", low_memory=False, error_bad_lines=False)
    df = get_series(logs_df)
    #print(df.head())
    #df = pd.read_csv("hourly_test.csv")
    #series = pd.Series(data=df["size"].values, index=pd.to_datetime(df["date_time"]))
    #series.plot()
    #print(series)
    #plt.show()
    
    df.to_csv(name + "_minute_max.csv")


main()