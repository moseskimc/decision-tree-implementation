import itertools

import pandas as pd
import numpy as np


def get_gini_impurity(y):
    """Compoute index of classification

        Args:
            y (pd.Series): classifcation values
    """

    # first check we have a series
    # if not throw an exception
    if not isinstance(y, pd.Series):
        raise Exception("Input must be a pd.Series object")

    # calculate individual probabilities (i.e. P_i)
    p = y.value_counts() / y.shape[0]

    # gini impurity
    gini = 1 - np.sum(p**2)

    return gini


def get_entropy(y):
    """Compute entropy given series y
    """

    # first check we have a series
    # if not throw an exception
    if not isinstance(y, pd.Series):
        raise Exception("Input must be a pd.Series object")

    # calculate individual probabilities (i.e. P_i)
    p = y.value_counts() / y.shape[0]

    # entropy
    epsilon = 1e-9  # small number
                    # to avoid taking log of zero
    entropy = np.sum(
        -p * np.log2(p + epsilon)  # broadcast
    )

    return entropy


def get_var(y):
    """Compute variance of random variable

        Args:
            y (pd.Series): random variable values
    """

    if len(y) == 1: return 0

    return y.var()


def get_ig(y, mask, impurity=get_entropy):
    """Compute IG for a single child split
    """

    prop = sum(mask) / len(mask)

    parent_impurity = impurity(y)

    child1_impurity = prop * impurity(y[mask])
    child2_impurity = (1 - prop) * impurity(y[~mask])

    ig = parent_impurity - (child1_impurity + child2_impurity)

    return ig


def generate_all_subsets(y):
    """Generate all possible subsets rand var values

    Args:
        y (pd.Series): random variable
    """

    y_unique = y.unique()

    subsets = []

    for subset_len in range(0, len(y_unique)+1):
        for subset in itertools.combinations(y_unique, subset_len):
            subset_list = list(subset)

            subsets.append(subset_list)

    return subsets


def get_best_split(target_label, data):
    
    ig_df = data.drop(
        "is_obese", axis=1
    ).apply(
        get_max_ig_split, y=data[target_label]
    )

    ig_df.rename(
        index=dict(
            zip(
                list(range(0,4)), ["max_ig", "max_ig_index", "best_split", "has_ig"]
            )
        ), inplace=True
    )

    # first check whether an ig has computed

    # take the last row and compute the sum

    if sum(ig_df.iloc[-1, :]) == 0: return None, None, None, None

    best_feature = max(ig_df)

    ig, split_index, split_value = ig_df[best_feature].values[:3]

    return best_feature, split_index, ig, split_value


def get_max_ig_split(x, y, impurity=get_entropy):
    """Compute the mask boundary with greatest information gain
    Args:
        x (pd.Series): predictor variable
        y (pd.Series): target variable
        impurity (func): entropy or variance

    """

    is_numeric = True if x.dtypes != "O" else False
    
    ig_values = []
    split_values = []


    if is_numeric:
        options = x.sort_values().unique()[1: ] # we skip the first
    # categorical
    else:
        options = generate_all_subsets(x)

    for option in options:
        mask = x < option if is_numeric else x.isin(option)

        # compute ig
        option_ig = get_ig(y, mask, impurity)

        ig_values.append(option_ig)
        split_values.append(option)
    # if there are no results

    if not ig_values:
        return None, None, None, False

    max_ig = max(ig_values)
    max_ig_index = ig_values.index(max_ig)
    best_split = split_values[max_ig_index]


    return max_ig, best_split, is_numeric, True


def make_split(variable, split_value, data, is_numeric):
    """Split data given boundary or subset
    """

    if is_numeric:
        mask = data[variable] < split_value
    else:
        mask = data[variable].isin(split_value)

    return data[mask], data[~mask]


def make_prediction(ser, is_numeric):

    if is_numeric: return ser.mean()

    return ser.value_counts().idxmax()


def pred_obs(observation, tree):
    """Output tree prediction on observation
    """

    # start at the root
    cond = list(tree.keys())[0]

    feature, split_type, split_val = cond.split()

    if split_type == "<=":
       # next, we check whether to traverse the further
       # check whether observations meets condition
       # splitting into subtrees
       answer = tree[cond][0] if observation[feature] <= float(split_val) else tree[cond][1]

    # if leaf, return value
    if not isinstance(answer, dict):
        return answer

    # traverse recursively
    return pred_obs(observation, answer)
