# %%
from mpl_toolkits.mplot3d import Axes3D
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, GridSearchCV
from pandas.plotting import scatter_matrix  # optional
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
%matplotlib inline

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
df.head()
# %%
X = df.iloc[:, [0, 1, 2, 3, 5, 6]]
y = df.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
# %%
