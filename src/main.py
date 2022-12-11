import os

import pandas as pd
import numpy as np

from src.decision_tree import DecisionTree


if __name__ == "__main__":
    
    # dir path
    path_to_dir = "./sample"
    filename = "500_Person_Gender_Height_Weight_Index.csv"

    filepath = os.path.join(path_to_dir, filename)
    
    # load data
    df = pd.read_csv(filepath)
    
    
    # create binary target variable "is_obese"
    df.loc[:, "is_obese"] = (df["Index"] >= 4).astype(int)
    
    # drop index column
    df.drop("Index", axis=1, inplace=True)
    
    
    # build tree
    max_depth = 5
    min_samples_split = 20
    min_information_gain  = 1e-5

    dt = DecisionTree(
        True,
        max_depth,
        min_samples_split,
        min_information_gain
    )
    
    # build tree
    dt.train(df, "is_obese")
    
    # train accuracy
    num_obs = df.shape[0]
    preds = []
    for i in range(num_obs):
        obs_pred = dt.predict(df.iloc[i,:])
        preds.append(obs_pred)

    acc = (np.array(preds) == df["is_obese"]).mean()

    print(f"train accuracy: {acc}")
