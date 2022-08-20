import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder


def fetch_species_class(string):

    # fetch folder name
    fold_name = string.split("\\")[-2]
    class_name = "_".join(fold_name.split(" ")[:-1])
    return class_name.replace("__", "_")  # added due to an anomaly of arousal of 2 underscores

def fetch_orientation_class(string):

    # fetch folder name
    fold_name = string.split("\\")[-2]
    return fold_name.split(" ")[-1]


def create_metadata():
    # fetching all the image files
    files = sorted(glob("dataset\\mosquito_dataset\\*\\*g"))

    # loading filenames int dataframe
    dataframe = pd.DataFrame({"path": files})

    # processing class names
    dataframe["species"] = dataframe["path"].apply(fetch_species_class)
    dataframe["sitting"] = dataframe["path"].apply(fetch_orientation_class)
    dataframe["path"] = dataframe["path"].apply(lambda x: "\\".join(x.split("\\")[-2:]))


    # performing encoding of classes
    for feature in ["species", "sitting"]:
        encoder = LabelEncoder()
        dataframe[f"{feature}_encoded"] = encoder.fit_transform(dataframe[feature])
    # storing metadata into cvs file
    dataframe.to_csv("dataset/metadata.csv", index=False)


if __name__ == "__main__":
    create_metadata()
