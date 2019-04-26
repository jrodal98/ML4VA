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
from src import config
from gmplot import gmplot
from src import k_means
%matplotlib inline
# %%
df = pd.read_csv("data/cleaned_data.csv")

# %%
# X = df.iloc[:, [1, 2, 3, 5, 6]]
df["ViolationDescription"].value_counts()
df.head()
X = df.iloc[:, [3, 5, 6, 7, 8]]
y = df.iloc[:, 4].apply(lambda x: 1 if x == "granted" else 0)
x_pipe = ColumnTransformer([
    ("numerical_vals", StandardScaler(), ["latitude", "longitude", "Hour"]),
    ("categorical_values", OneHotEncoder(), ["ViolationDescription", "DayIssued"])
])
X = x_pipe.fit_transform(X)
gps_scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# %%
dict_classifiers = {
    # "Logistic Regression": LogisticRegression(),
    # "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Non-Linear SVM": SVC(C=.01, gamma=.1, kernel="rbf"),
    # "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    # "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=800, min_samples_split=10, min_samples_leaf=4, max_features="sqrt", max_depth=50, bootstrap=True),
    # "Neural Net": MLPClassifier(alpha=1),
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
# awesome code for modeling http://ataspinar.com/2017/05/26/classification-with-scikit-learn/
dict_models = batch_classify(X_train, y_train, X_test, y_test, no_classifiers=3)
display_dict_models(dict_models)
# %%
# HYPERTUNING SVM


def plot_metric(metric, label, i, dolog=False):
    plt.subplot(2, 2, i)
    if dolog:
        plt.plot(np.log(c_values), metric)
        plt.xlabel("log(C)")
    else:
        plt.plot(c_values, metric)
        plt.xlabel("C")
    plt.title(f"{label} as a function of C")
    plt.ylabel("Score")

# %%


scoring_metrics = ["accuracy", "precision", "recall", "f1"]


def test_clfs(c):
    svm_clf = LinearSVC(C=c, loss="hinge", random_state=42, max_iter=1000)
    scores = cross_validate(svm_clf, X_train, y_train,
                            scoring=scoring_metrics, cv=3, return_train_score=False)
    return (np.mean(scores["test_accuracy"]), np.mean(scores["test_precision"]), np.mean(scores["test_recall"]), np.mean(scores["test_f1"]))


c_values = [.001, .01, .1, 1, 10, 100, 1000]
results = [test_clfs(c) for c in c_values]
accuracies, precisions, recalls, f1s = zip(*results)
# %%
fig = plt.figure(figsize=(10, 10))
plot_metric(accuracies, "Accuracy", 1, True)
plot_metric(precisions, "Precision", 2, True)
plot_metric(recalls, "Recall", 3, True)
plot_metric(f1s, "F1-Score", 4, True)
# plt.savefig("images/linear_svm.png")


# %%
best_acc = max(accuracies)
ind = accuracies.index(best_acc)
best_c = c_values[5]
print(best_c)
# %%
params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.0001, 0.001, 0.01, 0.1]}

grid_search = GridSearchCV(SVC(kernel="rbf"), params_grid,
                           scoring=scoring_metrics, refit=False, return_train_score=False)

grid_search.fit(X_train, y_train)
accuracies = list(grid_search.cv_results_["mean_test_accuracy"])
precisions = list(grid_search.cv_results_["mean_test_precision"])
recalls = list(grid_search.cv_results_["mean_test_recall"])
f1s = list(grid_search.cv_results_["mean_test_f1"])
nonlinear_c_values = list(grid_search.cv_results_["param_C"])
gamma_values = list(grid_search.cv_results_["param_gamma"])
# %%


def plot_3d_metric(ax, metric, label):
    ax.scatter(np.log(nonlinear_c_values), np.log(gamma_values), metric)
    ax.set_xlabel("log(C)")
    ax.set_ylabel("log(gamma)")
    ax.set_zlabel("Score")
    ax.set_title(f"{label} as a function of C and gamma")


fig = plt.figure(figsize=(12, 12))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
scores = (accuracies, precisions, recalls, f1s)

for i in range(1, 5):
    ax = fig.add_subplot(220 + i, projection='3d')
    plot_3d_metric(ax, scores[i-1], metrics[i-1])
plt.savefig("images/nonlinear_svm_hypertuning.png")
# %%
x = (accuracies, precisions, recalls, f1s, nonlinear_c_values, gamma_values)
categories = list(zip(*x))
categories.sort(key=lambda tup: (-tup[2]))
print(*categories[:10], sep="\n")
for i in categories:
    print(i)
    print("-------------------")
best_category = categories[5]
best_nonlinear_c = best_category[-2]
best_gamma = best_category[-1]
print(best_nonlinear_c, best_gamma)
# %%
non_linear_clf = SVC(C=.01, gamma=.1, kernel="rbf")
non_linear_clf.fit(X_train, y_train)
# %%


def test_svm(svm, X_test, y_test):
    predictions = svm.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy:{acc}\nPrecision:{precision}\nRecall:{recall}\nF1 Score:{f1}")


test_svm(non_linear_clf, X_test, y_test)


# %%
# gps_scaler = StandardScaler()
# gps_data = gps_scaler.fit_transform(df.iloc[:, [5, 6]])
k_m = k_means.K_means(X)
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
