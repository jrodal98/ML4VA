# %%
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import time
import seaborn as sns
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
from config import api_key
from gmplot import gmplot
from src import config
%matplotlib inline
# %%
df = pd.read_csv("data/cleaned_data.csv")

# %%
# X = df.iloc[:, [1, 2, 3, 5, 6]]
df["ViolationDescription"].value_counts()
df.head()
X = df.iloc[:, [3, 5, 6, 7]]
y = df.iloc[:, 4]
x_pipe = ColumnTransformer([
    ("numerical_vals", StandardScaler(), ["latitude", "longitude"]),
    ("categorical_values", OneHotEncoder(), ["ViolationDescription", "DayIssued"])
])
X = x_pipe.fit_transform(X)
gps_scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# %%
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha=1),
    # "Naive Bayes": GaussianNB()
}
# %%


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.

    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
    So it is best to train them on a smaller dataset first and
    decide whether you want to comment them out or not based on the test accuracy score.
    """

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.process_time()
        classifier.fit(X_train, Y_train)
        t_end = time.process_time()

        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {
            'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models


def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)), columns=[
        'classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    display(df_.sort_values(by=sort_by, ascending=False))


# %%
dict_models = batch_classify(X_train, y_train, X_test, y_test, no_classifiers=7)
display_dict_models(dict_models)


# %%


def dist(v1, v2):
    return (((v1 - v2)**2).sum(axis=2))**.5


class K_means:
    def __init__(self, values):
        self.values = values
        self.k = 0
        self.closest = None
        self.centroids = None

    def cluster(self, k):
        self.k = k
        centroids = np.random.permutation(self.values)[:k]  # Initialize centroid
        closest = np.empty(self.values.shape[0])  # Initialize the closest array
        previous_closest = None  # keep track of the previous closest array
        while not all(previous_closest == closest):  # while classes are still changing
            previous_closest = closest
            # populate an array with the closest centroid for each row in values
            closest = np.argmin(dist(self.values, centroids[:, np.newaxis]), axis=0)
            # new centroids are the mean point of the values assigned to that centroid
            centroids = np.array([self.values[closest == clust].mean(axis=0)
                                  for clust in range(centroids.shape[0])])
        self.centroids = centroids
        self.closest = closest
        return centroids

    def get_cluster_sds(self):
        return np.array([self.values[self.closest == clust].std(axis=0)
                         for clust in range(self.centroids.shape[0])])

    def squared_err(self):
        return np.array([(self.values[self.closest == clust]**2).sum(axis=0) - self.values[self.closest == clust].shape[0] * self.centroids[clust]**2 for clust in range(self.centroids.shape[0])])

    def get_clusters(self):
        return [self.values[self.closest == clust] for clust in range(self.centroids.shape[0])]


# %%
df["AppealStatus"].value_counts()
gmap.apikey = api_key
# gmap.scatter(df["latitude"], df["longitude"], '#FF0000', size=10, marker=False)
gmap.heatmap(df["latitude"], df["longitude"])
gmap.draw("my_map.html")
# awesome code for modeling http://ataspinar.com/2017/05/26/classification-with-scikit-learn/


# %%
gps_scaler = StandardScaler()
gps_data = gps_scaler.fit_transform(df.iloc[:, [5, 6]])
gps_data
k_m = K_means(gps_data)
centroids = k_m.cluster(8)
df["Cluster"] = k_m.closest

# %%
df.plot(kind="scatter", x="longitude", y="latitude",
        c="Cluster", cmap=plt.get_cmap("jet"), colorbar=False, figsize=(30, 30), alpha=.4)
# %%
gmap = gmplot.GoogleMapPlotter(df["latitude"].mean(), df["longitude"].mean(), 14)
gmap.apikey = api_key
# gmap.scatter(df["latitude"], df["longitude"], '#FF0000', size=10, marker=False)
colors = ["000000", "F0F8FF", "0000FF", "FF0000", "FF8C00", "006400", "FF00FF", "FFD700"]
for i, color in enumerate(colors):
    sub_df = df.where(df["Cluster"] == i).dropna()
    gmap.scatter(sub_df["latitude"], sub_df["longitude"], color, size=10, marker=False)

gmap.draw("my_map.html")
# gmap.html_color_codes
