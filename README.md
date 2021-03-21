# PolioCases
#Change
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from scipy import stats
import seaborn as sns

data = pd.read_csv('PolioHealth.csv', sep = ',')
data = data.drop('Year', axis = 1)

data.head()

#Convert status into numbers
status = pd.get_dummies(data.Status)
data = pd.concat([data, status], axis = 1)
data = data.drop(['Status'], axis=1)
data.rename(columns = {'Deloping' : '0', 'Developed' : 1})

#Mean of all the countries
data = data.groupby('Country').mean()
data.head()

data.columns

plt.scatter(data[' HIV/AIDS'], data['Life expectancy '])
plt.xlabel('HIV/AIDS')
plt.ylabel('Life expectancy')

plt.scatter(data.GDP, data['Life expectancy '])
plt.xlabel('GDP')
plt.ylabel('Life expectancy')

plt.scatter(data[' BMI '], data['Life expectancy '])
plt.xlabel('BMI')
plt.ylabel('Life expectancy')

plt.scatter(data['under-five deaths '], data['Life expectancy '])
plt.xlabel('under-five deaths')
plt.ylabel('Life expectancy')

plt.scatter(data['Alcohol'], data['Life expectancy '])
plt.xlabel('Alcohol')
plt.ylabel('Life expectancy')

plt.scatter(data['Adult Mortality'], data['Life expectancy '])
plt.xlabel('Adult Mortality')
plt.ylabel('Life expectancy')

plt.scatter(data['Schooling'], data['Life expectancy '])
plt.xlabel('Schooling')
plt.ylabel('Life expectancy')

plt.scatter(data['percentage expenditure'], data['Life expectancy '])
plt.xlabel('Percentage Healhcare expenditure')
plt.ylabel('Life expectancy')

plt.scatter(data['Polio'], data['Life expectancy '])
plt.xlabel('Polio')
plt.ylabel('Life expectancy')

plt.figure(figsize = (14, 10))
sns.heatmap(data.corr(), annot = True)

life_labels = data['Life expectancy ']
life_features = data.drop('Life expectancy ', axis = 1)

life_features.isnull().head()

life_features.isnull().sum()

life_labels.isnull().sum()

life_features.fillna(value = life_features.mean(), inplace = True)

life_labels.fillna(value = life_labels.mean(), inplace = True)

stats.describe(life_features[1:])

min_max_scaler = MinMaxScaler()
life_features = min_max_scaler.fit_transform(life_features)

life_features

life_features_train, life_features_test, life_labels_train, life_labels_test = train_test_split(
        life_features, life_labels, train_size = 0.7, test_size = 0.3)

#Linear Reg
linear_model = LinearRegression()
linear_model.fit(life_features_train, life_labels_train)

print('R_square score on the training: %.2f' % linear_model.score(life_features_train, life_labels_train))

linear_model_predict = linear_model.predict(life_features_test)

# Commented out IPython magic to ensure Python compatibility.
print('Coefficients: \n', linear_model.coef_)
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, linear_model_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, linear_model_predict))
print('R_square score: %.2f' % r2_score(life_labels_test, linear_model_predict))

# Commented out IPython magic to ensure Python compatibility.
#Ridge Reg
coring = make_scorer(r2_score)
grid_cv = GridSearchCV(Ridge(),
              param_grid={'alpha': range(0, 10), 'max_iter' : [10, 100, 1000]},
              scoring=scoring, cv=5, refit=True)
grid_cv.fit(life_features_train, life_labels_train)
print("Best Parameters: " + str(grid_cv.best_params_))
result = grid_cv.cv_results_
print("R^2 score on training data: %.2f" %grid_cv.score(life_features_train, life_labels_train))
print("R^2 score: %.2f"
#       % r2_score(life_labels_test, grid_cv.best_estimator_.predict(life_features_test)))
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, linear_model_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, linear_model_predict))

# Commented out IPython magic to ensure Python compatibility.
#Lasso Reg
scoring = make_scorer(r2_score)
grid_cv = GridSearchCV(Lasso(),
              param_grid={'alpha': range(0, 10), 'max_iter' : [10, 100, 1000]},
              scoring=scoring, cv=5, refit=True)
grid_cv.fit(life_features_train, life_labels_train)
print("Best Parameters: " + str(grid_cv.best_params_))
result = grid_cv.cv_results_
print("R^2 score on training data: %.2f" % grid_cv.score(life_features_train, life_labels_train))
print("R^2 score: %.2f"
#       % r2_score(life_labels_test, grid_cv.best_estimator_.predict(life_features_test)))
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, linear_model_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, linear_model_predict))

# Commented out IPython magic to ensure Python compatibility.
#Elastic Net
scoring = make_scorer(r2_score)
grid_cv = GridSearchCV(ElasticNet(),
param_grid={'alpha': range(0, 10), 'max_iter' : [10, 100, 1000], 'l1_ratio' : [0.1, 0.4, 0.8]},
scoring=scoring, cv=5, refit=True)
grid_cv.fit(life_features_train, life_labels_train)
print("Best Parameters: " + str(grid_cv.best_params_))
result = grid_cv.cv_results_
print("R^2 score on training data: %.2f" % grid_cv.score(life_features_train, life_labels_train))
print("R^2 score: %.2f"
#       % r2_score(life_labels_test, grid_cv.best_estimator_.predict(life_features_test)))
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, linear_model_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, linear_model_predict))

#Linear Reg with polynomial Features
quad_feature_transformer = PolynomialFeatures(2, interaction_only = True)
quad_feature_transformer.fit(life_features_train)
life_features_train_quad = quad_feature_transformer.transform(life_features_train)
life_features_test_quad = quad_feature_transformer.transform(life_features_test)

poly_model_quad = LinearRegression()
poly_model_quad.fit(life_features_train_quad, life_labels_train)
accuracy_score_quad = poly_model_quad.score(life_features_train_quad, life_labels_train)
print(accuracy_score_quad)

poly_model_quad_predict = poly_model_quad.predict(life_features_test_quad)

# Commented out IPython magic to ensure Python compatibility.
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, poly_model_quad_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, poly_model_quad_predict))
print('R_square score: %.2f' % r2_score(life_labels_test, poly_model_quad_predict))

#Decision Tree
decision_tree_model = DecisionTreeRegressor()
decision_tree_fit = decision_tree_model.fit(life_features_train, life_labels_train)
decision_tree_score=cross_val_score(decision_tree_fit, life_features_train, life_labels_train, cv = 5)
print("mean cross validation score: %.2f"  % np.mean(decision_tree_score))
print("score without cv: %.2f" % decision_tree_fit.score(life_features_train, life_labels_train))
print("R^2 score on the test data %.2f"% r2_score(life_labels_test, decision_tree_fit.predict(life_features_test)))

decision_tree_model_predict = decision_tree_model.predict(life_features_test)

# Commented out IPython magic to ensure Python compatibility.
scoring = make_scorer(r2_score)
grid_cv = GridSearchCV(DecisionTreeRegressor(),
              param_grid={'min_samples_split': range(2, 10)},
              scoring=scoring, cv=5, refit=True)
grid_cv.fit(life_features_train, life_labels_train)
grid_cv.best_params_
print('Best Parameters:'+str(grid_cv.best_params_))
result = grid_cv.cv_results_
print("R^2 score on training data: %.2f"  % grid_cv.best_estimator_.score(life_features_train, life_labels_train))
print("R^2 score: %.2f"
#       % r2_score(life_labels_test, grid_cv.best_estimator_.predict(life_features_test)))
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, decision_tree_model_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, decision_tree_model_predict))

# Commented out IPython magic to ensure Python compatibility.
#Random Forest
random_forest_model = RandomForestRegressor()
random_forest_fit = random_forest_model.fit(life_features_train, life_labels_train)
random_forest_score = cross_val_score(random_forest_fit, life_features_train, life_labels_train, cv = 5)
print("mean cross validation score: %.2f"
#        % np.mean(random_forest_score))
print("score without cv: %.2f"
#       % random_forest_fit.score(life_features_train, life_labels_train))
print("R^2 score on the test data %.2f"
#       %r2_score(life_labels_test, random_forest_fit.predict(life_features_test)))

random_forest_model_predict = random_forest_model.predict(life_features_test)

# Commented out IPython magic to ensure Python compatibility.
scoring = make_scorer(r2_score)
grid_cv = GridSearchCV(RandomForestRegressor(),
              param_grid={'min_samples_split': range(2, 10)},
              scoring=scoring, cv=5, refit=True)
grid_cv.fit(life_features_train, life_labels_train)
grid_cv.best_params_
result = grid_cv.cv_results_
print("Best Parameters: " + str(grid_cv.best_params_))
result = grid_cv.cv_results_
print("R^2 score on training data: %.2f"  % grid_cv.best_estimator_.score(life_features_train, life_labels_train))
print("R^2 score: %.2f"
#       % r2_score(life_labels_test, grid_cv.best_estimator_.predict(life_features_test)))
print("Mean squared error: %.2f"
#       % mean_squared_error(life_labels_test, random_forest_model_predict))
print("Mean absolute error: %.2f"
#       % mean_absolute_error(life_labels_test, random_forest_model_predict))
