from src.utils import make_split, get_best_split, make_prediction, pred_obs


class DecisionTree:
    
    def __init__(
        self,
        is_target_categorical,
        max_depth=None,
        min_samples_split=None,
        min_information_gain=1e-10,
        max_categories=20
    ):
        
        self.is_target_categorical = is_target_categorical
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.max_categories = max_categories
        
        self.data = None
        self.target_var = None
        self.tree = None
    
    
    def build_tree(self, data, counter=0):
        
        # check max categories
        if counter==0:  # once a split happens, depth increases by 1 and max cat checked
            types = self.data.dtypes
            check_columns = types[types=="object"].index

            cat_df = self.data[check_columns]
            unique_val_ser = cat_df.apply(lambda x: len(x.unique()))

            if sum(unique_val_ser > self.max_categories) > 0:
                raise Exception("Number of categories is too many!")

        # check max depth
        if self.max_depth == None:
            depth_cond = True
        else: # check counter
            depth_cond = True if counter < self.max_depth else False
        
        # check min samples
        if self.min_samples_split == None:
            sample_cond = True
        else:
            sample_cond = True if data.shape[0] > self.min_samples_split else False

        
        # if depth and min_samples are a check
        # check ig condition is met

        if depth_cond and sample_cond:
            var, split_val, ig, is_numeric = get_best_split(self.target_var, data)

            # check ig condition
            if ig is not None and ig >= self.min_information_gain:
                counter += 1 # depth increase
                # split data
                left, right = make_split(var, split_val, data, is_numeric)

                # create a subtree
                split_type = "<=" if is_numeric else "in"
                condition = f"{var} {split_type}  {split_val}"
                subtree = {condition: []}

                # find splits/leaves (via recursion)
                obese_pos = self.build_tree(
                    left,
                    counter
                )

                obese_neg = self.build_tree(
                    right,
                    counter
                )

                # the only time obese_pos equals obese_neg
                # is when you have already reached a leaf
                # and there is a prediction
                if obese_pos == obese_neg: 
                    subtree = obese_pos
                # otherwise, we have subtrees
                else:
                    subtree[condition].append(obese_pos)
                    subtree[condition].append(obese_neg)
            # found a leaf
            else:
                return make_prediction(
                    data[self.target_var],
                    not self.is_target_categorical
                )
        # depth or min_samples condition is not met
        else:
            return make_prediction(
                    data[self.target_var],
                    not self.is_target_categorical
                )

        return subtree
    
    def train(self, data, target_var):
        
        self.data = data
        self.target_var = target_var
        
        self.tree = self.build_tree(data)
    
    def predict(self, observation):
        """Output tree prediction on observation
        """
        
        prediction = pred_obs(observation, self.tree)
        
        return prediction
