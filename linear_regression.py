import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt 

np.random.seed(34)

# generate synthetic data
x = 2 * np.random.rand(100, 1)  # generate 100 random numbers between 0 & 2
y = 4 + 4 * x + np.random.rand(100, 1) # linear relationship with noise

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

mse = mean_squared_error(y_test, y_prediction)

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, y_prediction, color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.title(mse)
plt.show()

#print('Mean Squared Error:',mse)