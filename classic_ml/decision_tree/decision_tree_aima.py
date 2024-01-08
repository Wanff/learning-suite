#%%
import numpy as np
import pandas as pd

# df = pd.read_csv("obesity_db.csv")

# df['obese'] = (df.Index >= 4).astype('int')
# df.drop('Index', axis = 1, inplace = True)

# data = df.to_numpy()

#%
df = pd.read_csv("in-vehicle-coupon-recommendation.csv")

df = df[["destination", "passanger", "weather", "temperature", "time", "coupon", "expiration", "Y"]]

# df
# df['obese'] = (df.Index >= 4).astype('int')
# df.drop('Index', axis = 1, inplace = True)

data = df.to_numpy()

#%%
df = pd.read_csv("toy_ds.txt", sep = " ")

df.drop("Ex", axis = 1, inplace = True)
df

#%%
data = df.to_numpy()

# %%
import random 

def calc_cat_to_freq(x: np.array):
    cat_to_freq = {}
    
    for el in x:
        if el not in cat_to_freq.keys():
            cat_to_freq[el]= 0
        
        cat_to_freq[el] += 1
    
    for cat in cat_to_freq:
        cat_to_freq[cat] /= x.shape[0]
    
    return cat_to_freq

def plurality(exs):
    """_summary_

    Args:
        exs (_type_): _description_
    """
    cat_to_freq = calc_cat_to_freq(exs[:, -1])
    
    #makes plurality random since max always gets first element
+ 13    shuffled_keys = sorted((list(cat_to_freq.keys())), key = lambda x: random.random())
    
    shuffled_cat_to_freq = [(cat, cat_to_freq[cat]) for cat in shuffled_keys]
    
    return max(shuffled_cat_to_freq, key = lambda x : x[1])[0]

def entropy(x: np.array):
    cat_to_freq = calc_cat_to_freq(x)
    
    entropy = sum([ -cat_to_freq[cat] * np.log2(cat_to_freq[cat] + 1e-09) for cat in cat_to_freq])

    return entropy

def importance(a, labels):
    cat_to_labels = {} 
    
    for i, cat in enumerate(a):
        if cat not in cat_to_labels:
            cat_to_labels[cat] = []
            
        cat_to_labels[cat].append(labels[i])
    
    num_dp = labels.shape[0]
    gain = entropy(labels) - sum([ (len(cat_to_labels[cat]) / num_dp) *  entropy(np.array(cat_to_labels[cat])) for cat in cat_to_labels ])
    
    return gain

def decision_tree_learning(exs, attrs, par_exs):
    """generates a decision tree

    Args:
        exs (np.array): the datapoints where the last index datapoint is the class
        attr (np.array): indices of the attributes
        par_exs (_type_): the parent datapoints
    """
    if exs.size == 0:
        return plurality(par_exs)
    elif len(set(exs[:, -1].tolist())) == 1:
        #if all examples have same classification
        return exs[:, -1][0]
    elif len(attrs) == 0:
        return plurality(exs)
    else:
        labels = exs[:, -1]
        importances = [importance(exs[:, int(a[0])], labels) for a in attrs]
        
        imp_attr_idx = importances.index(max(importances))
        
        most_important_a = attrs[imp_attr_idx][1]
        tree = {most_important_a: {}}
        print("outside")
        print(tree)
        
        imp_attr_data_idx = int(attrs[imp_attr_idx][0])

        for v in set( par_exs[:,imp_attr_data_idx ].tolist() ):

            v_idxs = np.argwhere(exs[:, imp_attr_data_idx] == v).squeeze(1)
            
            new_exs = exs[v_idxs]

            new_attrs = np.delete(attrs, imp_attr_idx, 0)

            subtree = decision_tree_learning(new_exs, new_attrs, exs)

            tree[ most_important_a ][v] = subtree
            
            print(tree)
        return tree
        

#%%
attrs = np.array(list(zip(list( range(data.shape[1]- 1)), df.columns[:-1])))


tree = decision_tree_learning(data, attrs, data)


#%%
tree

# %%
np.delete(attrs, 4, 0)
# %%

# %%
from sklearn import tree

one_hot_data = pd.get_dummies(df[df.columns[:-1]],drop_first=True)


labels = [1 if d == "Yes" else 0 for d in data[:, -1]]
one_hot_data

#%%
clf = tree.DecisionTreeClassifier().fit(one_hot_data, labels)


# %%
tree.plot_tree(clf)
# %%
a = [(1, 2), (3, 4), (2, 6)]

max(a, key = lambda x: x[0])
# %%
a = random.shuffle(["Yes", "No"])
print(a)


# %%
# %%

a = [1, 2, 3,4 , 5, 6]

a[:-1]
# %%


a = [1, 2, 3, 4, 5]

#%%
any([False, True])
# %%
ys = [1, 2, 3, 10]

num_labels = 4

all([y in list(range(num_labels)) for y in ys])
# %%
