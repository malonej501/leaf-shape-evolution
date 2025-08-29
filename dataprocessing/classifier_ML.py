"""Machine learning classifier - not used for the final paper."""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import re

wd = "/home/m/malone/vlab/oofs/ext/NPHLeafModels_1.01/LeafGenerator"
wd1 = "/home/m/malone/leaf_storage/random_walks"


def classify_test():

    # actual data
    labeled_data = pd.read_csv(
        "/home/m/malone/leaf_storage/accuracy_test_500/trainingdata_morph.csv")

    unlabeled_data = pd.read_csv(
        "/home/m/malone/leaf_storage/accuracy_test_200/shape_report.csv")
    # unlabeled_data["shape"] = None
    # print(unlabeled_data)

    labeled_features = labeled_data[["no.extrema", "minmax_dist_avg",
                                     "minmax_angle_avg", "minima_samerow_dist_avg", "refmax_dist_avg"]].values
    labeled_labels = labeled_data["shape"].values

    # Convert the labels to numerical values
    label_encoder = LabelEncoder()
    labeled_labels = label_encoder.fit_transform(labeled_labels)

    # Split the labeled set into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        labeled_features, labeled_labels, random_state=42
    )

    # Train a decision tree classifier on the labeled training set
    clf = HistGradientBoostingClassifier()
    clf.fit(train_features, train_labels)

    # Predict labels for the validation features
    # val_pred_labels = clf.predict(val_features)
    # accuracy = accuracy_score(val_labels, val_pred_labels)
    # print(f"Accuracy on validation set: {accuracy}")

    # val_pred_labels = label_encoder.inverse_transform(val_pred_labels)
    # val_labels = label_encoder.inverse_transform(val_labels)
    # for i, val_label in enumerate(val_labels):
    #     print(val_label, val_pred_labels[i])

    # # Use the trained classifier to predict labels for the unlabeled set
    unlabeled_features = unlabeled_data[["no.extrema", "minmax_dist_avg",
                                         "minmax_angle_avg", "minima_samerow_dist_avg", "refmax_dist_avg"]].values
    predicted_labels = clf.predict(unlabeled_features)

    # Convert the predicted numerical labels back into their original form
    predicted_labels = label_encoder.inverse_transform(predicted_labels)

    # print(predicted_labels)  # ['foo', 'bar']

    unlabeled_data["predicted"] = predicted_labels
    # print(unlabeled_data)

    unlabeled_data.to_csv(
        "/home/m/malone/leaf_storage/accuracy_test_200/prediction.csv", index=False)


def classify_batch(wd, wd1):

    dfs = []

    for leafdirectory in os.listdir(wd1 + "/leaves_full_28-8-23_MUT1"):
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(
            wd1 + "/leaves_full_28-8-23_MUT123", leafdirectory)
        for walkdirectory in os.listdir(leafdirectory_path):
            walkdirectory_path = os.path.join(
                leafdirectory_path, walkdirectory)
            for file in os.listdir(walkdirectory_path):
                if file.endswith(".csv") and "shape_report" in file:
                    df = pd.read_csv(os.path.join(walkdirectory_path, file))
                    df.insert(0, "leafid", leafdirectory)
                    df.insert(1, "walkid", int(
                        re.findall(r"\d+", walkdirectory)[0]))
                    dfs.append(df)

    unlabeled_data = pd.concat(dfs, ignore_index=True)
    unlabeled_data.to_csv(wd1 + "/unlabeled_data.csv", index=False)
    print("#### Concatenation done")

    # actual data
    labeled_data = pd.read_csv(wd + "/trainingdata/shape_report.csv")

    labeled_features = labeled_data[["no.extrema", "minmax_dist_avg",
                                     "minmax_angle_avg", "minima_samerow_dist_avg", "refmax_dist_avg"]].values
    labeled_labels = labeled_data["shape"].values

    # Convert the labels to numerical values
    label_encoder = LabelEncoder()
    labeled_labels = label_encoder.fit_transform(labeled_labels)

    # Split the labeled set into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        labeled_features, labeled_labels, test_size=0.2, random_state=42
    )

    # Train a decision tree classifier on the labeled training set
    clf = HistGradientBoostingClassifier()
    clf.fit(train_features, train_labels)
    print("#### Training done")

    # Use the trained classifier to predict labels for the unlabeled set
    unlabeled_features = unlabeled_data[["no.extrema", "minmax_dist_avg",
                                         "minmax_angle_avg", "minima_samerow_dist_avg", "refmax_dist_avg"]].values
    predicted_labels = clf.predict(unlabeled_features)

    # Convert the predicted numerical labels back into their original form
    predicted_labels = label_encoder.inverse_transform(predicted_labels)
    print("#### Prediction done")

    # print(predicted_labels)  # ['foo', 'bar']

    unlabeled_data["predicted"] = predicted_labels
   # print(unlabeled_data)

    unlabeled_data.to_csv(wd1 + "/prediction.csv", index=False)


def classify_test_contour():

    # actual data
    # labeled_data = pd.read_csv(wd + "/trainingdata/trainingdata.csv", header=None)
    labeled_data = pd.read_csv(
        "/home/m/malone/leaf_storage/accuracy_test_500/trainingdata_contour.csv", header=None)

    unlabeled_data = pd.read_csv(
        "/home/m/malone/leaf_storage/accuracy_test_200/refdist_report.csv", header=None)
    # #unlabeled_data["shape"] = None
    # print(unlabeled_data)

    labeled_features = labeled_data.iloc[:, 2:].values
    labeled_labels = labeled_data.iloc[:, 1].values

    # Convert the labels to numerical values
    label_encoder = LabelEncoder()
    labeled_labels = label_encoder.fit_transform(labeled_labels)

    # Split the labeled set into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        labeled_features, labeled_labels, random_state=42
    )

    # Train a decision tree classifier on the labeled training set
    clf = HistGradientBoostingClassifier()
    clf.fit(train_features, train_labels)

    # Predict labels for the validation features
    # val_pred_labels = clf.predict(val_features)
    # accuracy = accuracy_score(val_labels, val_pred_labels)
    # print(f"Accuracy on validation set: {accuracy}")
    # val_pred_labels = label_encoder.inverse_transform(val_pred_labels)

    # # Use the trained classifier to predict labels for the unlabeled set
    unlabeled_features = unlabeled_data.iloc[:, 1:].values
    predicted_labels = clf.predict(unlabeled_features)

    # # Convert the predicted numerical labels back into their original form
    predicted_labels = label_encoder.inverse_transform(predicted_labels)

    # print(predicted_labels)  # ['foo', 'bar']

    unlabeled_data["predicted"] = predicted_labels
    # print(unlabeled_data)

    unlabeled_data.to_csv(
        "/home/m/malone/leag_storage/accuracy_test_200/prediction.csv", index=False)


def classify_batch_contour(wd, wd1):

    dfs = []

    for leafdirectory in os.listdir(wd1 + "/leaves_full_28-8-23_MUT1"):
        print(f"Current = {leafdirectory}")
        leafdirectory_path = os.path.join(
            wd1 + "/leaves_full_28-8-23_MUT1", leafdirectory)
        for walkdirectory in os.listdir(leafdirectory_path):
            walkdirectory_path = os.path.join(
                leafdirectory_path, walkdirectory)
            for file in os.listdir(walkdirectory_path):
                if file.endswith(".csv") and "refdist_report" in file:
                    df = pd.read_csv(os.path.join(walkdirectory_path, file))
                    df.insert(0, "leafid", leafdirectory)
                    df.insert(1, "walkid", int(
                        re.findall(r"\d+", walkdirectory)[0]))
                    df.columns = range(df.shape[1])
                    dfs.append(df)

    widest_df = max(dfs, key=len)
    filled_dfs = [df.reindex(columns=widest_df.columns,
                             fill_value=np.nan) for df in dfs]

    unlabeled_data = pd.concat(filled_dfs, axis=0, ignore_index=True)
    unlabeled_data.to_csv(wd1 + "/unlabeled_data.csv",
                          index=False, header=False)

    # actual data
    labeled_data = pd.read_csv(wd + "/trainingdata/refdist_report.csv")

    labeled_features = labeled_data.iloc[:, 2:].values
    labeled_labels = labeled_data.iloc[:, 1].values

    # Convert the labels to numerical values
    label_encoder = LabelEncoder()
    labeled_labels = label_encoder.fit_transform(labeled_labels)

    # Split the labeled set into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        labeled_features, labeled_labels, test_size=0.2, random_state=42
    )

    # Train a decision tree classifier on the labeled training set
    clf = HistGradientBoostingClassifier()
    clf.fit(train_features, train_labels)

    print("Training Done")

    # Use the trained classifier to predict labels for the unlabeled set
    unlabeled_features = unlabeled_data.iloc[:, 3:].values
    predicted_labels = clf.predict(unlabeled_features)

    print("Prediction Done")

    # Convert the predicted numerical labels back into their original form
    predicted_labels = label_encoder.inverse_transform(predicted_labels)

    print(predicted_labels)  # ['foo', 'bar']

    prediction = pd.DataFrame(
        {"leafid": unlabeled_data.iloc[:, 0], "walkid": unlabeled_data.iloc[:, 1], "predicted": predicted_labels})

    prediction.to_csv(wd1 + "/prediction.csv", index=False)


# classify_batch_contour(wd, wd1)
classify_test_contour()
# classify_test()
# classify_batch(wd,wd1)
