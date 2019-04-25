# %%
import pandas as pd
import numpy as np
from config import api_key
from geopy.geocoders import GoogleV3
# %%
df = pd.read_csv("ticket_data.csv")
df.dropna(inplace=True)
df.head()

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
df["DateIssued"] = df["DateIssued"].apply(lambda x: x[:10] if int(x[:2]) <= 21 else np.nan)
df["TimeIssued"] = df["TimeIssued"].apply(fix_time)
df["DateIssued"] = pd.to_datetime(df["DateIssued"])
df["DayIssued"] = df["DateIssued"].dt.weekday_name
df["ViolationDescription"] = df["ViolationDescription"].apply(lambda x: x.strip())

# %%
df = df.where((df["latitude"] < 38.4) & (df["AppealStatus"] != "pending"))
df.dropna(inplace=True)

# # %%
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

# %%
df.to_csv("cleaned_data.csv", index=False)
