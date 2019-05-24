import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt


# Get specific data from CSV file
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors']]
Y = data['price']

# Get training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1 / 3, random_state = 0)

# Format training and testing data
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)

# Train the model by xtrain and ytrain
model = LinearRegression()
model.fit(xtrain, ytrain)

# Show coefficient of each house feature
# pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
pd.DataFrame(
    {
        "Feature" : X.columns,
        "Coefficient" : np.transpose(model.coef_)
    }
)

# Show intercepts of house feature data
# Show intercepts of house feature data
model.intercept_

# Predict house price.
# Parameters are number of bedrooms, number of bathrooms, house square feet, and number of storeys
print(model.predict([[3, 2, 2500, 2]])[0]) # round(num,2)

# Mean squared error on training data
pred = model.predict(xtrain)
((pred - ytrain) * (pred - ytrain)).sum() / len(ytrain)

# Mean squared error on testing data
predtest = model.predict(xtest)
((predtest - ytest) * (predtest - ytest)).sum() / len(ytest)

# Average relative deviation on training data
(abs(pred - ytrain) / ytrain).sum() / len(ytrain)

# Average relative deviation on testing data
(abs(predtest - ytest) / ytest).sum() / len(ytest)

# xtrain, xtest, ytrain, ytest
def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #a、b、r
    print("R squared：", r_value ** 2)

# rsquared(data['bedrooms'], data['price'])
# rsquared(data['bathrooms'], data['price'])
# rsquared(data['sqft_living'], data['price'])
# rsquared(data['floors'], data['price'])

# # Show relationship between house size and selling price
# plt.scatter(X['sqft_living'], Y)
# plt.title('Relationship Between House Size And Selling Price')
# plt.xlabel('Square Feet')
# plt.ylabel('Dollar')
# plt.show()

# # house size counts
# X['sqft_living'].hist()
# plt.show()

# boundary values
# print("max bedrooms: " + str(int(max(data["bedrooms"]))))
# print("min bedrooms: " + str(int(min(data["bedrooms"]))))
#
# print("max bathrooms: " + str(int(max(data["bathrooms"]))))
# print("min bathrooms: " + str(int(min(data["bathrooms"]))))
#
# print("max sqft_living: " + str(max(data["sqft_living"])))
# print("min sqft_living: " + str(min(data["sqft_living"])))
#
# print("max floors: " + str(max(data["floors"])))
# print("min floors: " + str(min(data["floors"])))
#
# print("max price: " + str(max(data["price"])))
# print("min price: " + str(min(data["price"])))