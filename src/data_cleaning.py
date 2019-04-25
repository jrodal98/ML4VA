# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("data/ticket_data.csv")
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
    t = f"{'0' * max(5-len(time),0)}{time}:00"
    if int(t[:2]) < 1 or int(t[:2]) > 24:
        t = np.nan
    return t


# %%
df["DateIssued"] = df["DateIssued"].apply(lambda x: x[:10] if int(x[:2]) <= 21 else np.nan)
df["TimeIssued"] = df["TimeIssued"].apply(fix_time)
df["DateIssued"] = pd.to_datetime(df["DateIssued"])
df["DayIssued"] = df["DateIssued"].dt.weekday_name
df["ViolationDescription"] = df["ViolationDescription"].apply(lambda x: x.strip())

# %%
df = df.where((df["latitude"] < 38.4) & (df["AppealStatus"] != "pending"))
df.dropna(inplace=True)
df["Hour"] = df["TimeIssued"].apply(lambda x: int(x[:2]))

# %%
df.to_csv("data/cleaned_data.csv", index=False)


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
