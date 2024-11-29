import ssl
import pandas as pd
import certifi
import requests

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location", "wheel_base",
           "length", "width", "height", "curb_weight", "engine_type", "num_cylinders",
           "engine_size", "fuel_system", "bore", "stroke", "compression_ratio",
           "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]

response = requests.get(url, verify=certifi.where())
if response.status_code == 200:
    with open("autos.csv", "w") as file:
        file.write(response.text)
    data = pd.read_csv("autos.csv", names=columns, na_values="?")
else:
    print("Ошибка при загрузке данных:", response.status_code)
    exit()

print(data.head())
