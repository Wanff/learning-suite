#%%
import pandas as pd
import numpy as np

df = pd.read_csv("obesity_db.csv")

df['obese'] = (df.Index >= 4).astype('int')
df.drop('Index', axis = 1, inplace = True)

data = df.to_numpy()
#%%
def calc_cat_to_freq(x: np.array):
    cat_to_freq = {}
    
    for el in x:
        if el not in cat_to_freq.keys():
            cat_to_freq[el]= 0
        
        cat_to_freq[el] += 1
    
    for cat in cat_to_freq:
        cat_to_freq[cat] /= x.shape[0]
    
    return cat_to_freq
# %%

def pd_gini_index(y):
  '''
  Given a Pandas Series, it calculates the Gini Impurity. 
  y: variable with which calculate Gini Impurity.
  '''
  if isinstance(y, pd.Series):
    p = y.value_counts()/y.shape[0]
    gini = 1-np.sum(p**2)
    return(gini)

  else:
    raise('Object must be a Pandas Series.')

def gini_index(x: np.array):
    cat_to_freq = calc_cat_to_freq(x)
    
    gini_index = 1 - sum([cat_to_freq[cat]**2 for cat in cat_to_freq])
    
    return gini_index

print(pd_gini_index(df.Gender))

print(gini_index(df.Gender.to_numpy()))

#%%

# %%
def pd_entropy(y):
  '''
  Given a Pandas Series, it calculates the entropy. 
  y: variable with which calculate entropy.
  '''
  if isinstance(y, pd.Series):
    a = y.value_counts()/y.shape[0]
    entropy = np.sum(-a*np.log2(a+1e-9))
    return(entropy)

  else:
    raise('Object must be a Pandas Series.')

def entropy(x: np.array):
    cat_to_freq = calc_cat_to_freq(x)
    
    entropy = sum([ -cat_to_freq[cat] * np.log2(cat_to_freq[cat] + 1e-09) for cat in cat_to_freq])

    return entropy

print(pd_entropy(df.Gender))
print(entropy(df.Gender.to_numpy()))

# %%
def variance(y):
  '''
  Function to help calculate the variance avoiding nan.
  y: variable to calculate variance to. It should be a Pandas Series.
  '''
  if(len(y) == 1):
    return 0
  else:
    return y.var()

def pd_information_gain(y, mask, func=pd_entropy):
  '''
  It returns the Information Gain of a variable given a loss function.
  y: target variable.
  mask: split choice.
  func: function to be used to calculate Information Gain in case os classification.
  '''
  
  a = sum(mask)
  b = mask.shape[0] - a
  
  if(a == 0 or b ==0): 
    ig = 0
  
  else:
    # if y.dtypes != 'O':
    #     print("regres")
    #     ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
    # else:
        # print("class")
    ig = func(y)-a/(a+b)*func(y[mask])-b/(a+b)*func(y[-mask])
  
  return ig

def information_gain(x: np.array, y: np.array):
    cat_to_labels = {} 
    
    for i, cat in enumerate(x):
        if cat not in cat_to_labels:
            cat_to_labels[cat] = []
            
        cat_to_labels[cat].append(y[i])
    
    num_dp = y.shape[0]
    gain = entropy(y) - sum([ (len(cat_to_labels[cat]) / num_dp) *  entropy(np.array(cat_to_labels[cat])) for cat in cat_to_labels ])
    
    return gain

print(pd_information_gain(df['obese'], df['Gender'] == 'Male'))
print(information_gain(df.Gender.to_numpy(), df.obese.to_numpy()))
    
# %%
def best_split_numeric(x: np.array, y: np.array):
    split_options = sorted(set(x.tolist()))[1:]
    
    gains = []
    for split_option in split_options:
        split_data = [1 if el >= split_option else 0 for el in x]
        gains.append(information_gain(split_data, y))
    
    return split_options[gains.index(max(gains))], max(gains)

best_split_numeric(df.Weight.to_numpy(), df.obese.to_numpy())

def best_split_categorical(x: np.array, y: np.array):
  
  




# %%


# %%
