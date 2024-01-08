#%%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import plotly.graph_objects as go

#%%
# Generate 2D data with means close to each other and high variance
mean1 = [5, 5]
mean2 = [-5, 1.1]
cov = [[10, 7], [7, 10]]
data = np.random.multivariate_normal(mean1, cov, 1000)
data2 = np.random.multivariate_normal(mean2, cov, 1000)

X = np.concatenate((data, data2))
y = np.concatenate((np.zeros(1000), np.ones(1000)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y)))

# Fit the model
model = LogisticRegression(fit_intercept=True)
model.fit(X, y)

# Plot the line fitted by the model
x_line = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_line = -(model.coef_[0][0] * x_line + model.intercept_) / model.coef_[0][1]
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Fitted Line'))

fig.show()

# Print accuracy score
print(accuracy_score(y, model.predict(X)))

#%%
from sklearn.decomposition import PCA

# Fit PCA on the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the first two principal components
fig_pca = go.Figure()
fig_pca.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers', marker=dict(color=y)))
fig_pca.show()

# %%
pca.explained_variance_ratio_
# %%
