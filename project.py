# %%
from config import api_key
from geopy.geocoders import GoogleV3
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import api_key

# %%


def fix_time(time):
    time = str(time).strip().replace(".", "")
    if len(time) == 1:
        time = "0" + time
    if ":" not in time:
        if (len(time)) == 5:
            time = time[:-1]
        time = time[:-2] + ":" + time[-2:]
    return f"{'0' * max(5-len(time),0)}{time}:00"


# %%
df = pd.read_csv("Parking_Tickets.csv")
df = df.iloc[:, [13, 2, 4, 11, 12]]
df.head()

# %%
gps = pd.read_csv("GPS_data.csv")
d = {}
for index, row in gps.iterrows():
    d[row["Location"]] = (row["latitude"], row["longitude"])
d
# %%
df["latitude"] = np.nan
df["longitude"] = np.nan
for index, row in df.iterrows():
    data = d[row["Location"]]
    df.iloc[index, 5] = data[0]
    df.iloc[index, 6] = data[1]
df.head()
# %%
x = df.copy()
x = x.iloc[:, :7]
x.to_csv("ticket_data.csv",index=False)
# %%
c = Counter(df["Location"])
c.most_common()

# %%

# g = GoogleV3(api_key)
# locations = df["Location"].unique()
# locations = pd.DataFrame(locations, columns=["Location"])
# locations["latitude"] = np.nan
# locations["longitude"] = np.nan
# locations.head()
# # %%
# current_index = 0
# while current_index < locations.shape[0]:
#     try:
#         coords = g.geocode(locations.iloc[current_index, 0].strip() + ", Charlottesville, VA")
#         if coords:
#             locations.iloc[current_index, 1] = coords.latitude
#             locations.iloc[current_index, 2] = coords.longitude
#         current_index += 1
#         print(f"{current_index}/{locations.shape[0]}")
#     except:  # timeout exception, don't remember what it is.
#         print(f"Error on the access for {locations.iloc[current_index, 0]}.  Trying again.")
#
# locations.to_csv("GPS_data.csv")
#
# # %%
# plt.scatter(locations["latitude"], locations["longitude"])

# %%
df.describe()

# %%
df["AppealStatus"].fillna("No Appeal", inplace=True)
df["DateIssued"] = df["DateIssued"].apply(lambda x: x[:10] if int(x[:2]) <= 21 else np.nan)
df["TimeIssued"] = df["TimeIssued"].apply(fix_time)
df.dropna(inplace=True)

# %%
df.info()  # there's only like 22 null rows, so I'll just drop them
df.head()
# pd.to_datetime(df["DateIssued"] + " " + df["TimeIssued"]).head()
s = set()
for time in df["TimeIssued"]:
    try:
        if int(time[3:5]) > 59:
            s.add(time)
    except:
        s.add(time)
for val in list(s):
    print(val)
