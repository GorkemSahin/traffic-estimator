import pandas as pd
import matplotlib.pyplot as plt

file_names = [
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

series = pd.Series(dtype=int, name="size");

for name in file_names:
    df = pd.read_csv(name + "_hourly.csv")
    monthly_series = pd.Series(data=df["size"].values, name="size", index=pd.to_datetime(df["date_time"]))
    print(monthly_series.head())
    series = series.append(monthly_series)

series.to_csv("all_hourly.csv")