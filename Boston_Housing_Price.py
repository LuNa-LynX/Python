import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
import seaborn as sns 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 
df = pd.read_csv(r"C:\Users\Swathi Krishna\Downloads\boston.csv") 
df.head() 
x = df[['LSTAT']] 
y = df['MEDV'] 
plt.scatter(x=x, y=y) 
plt.title("LSTAT vs MEDV") 
plt.xlabel("Average of lower status of the population (LSTAT)") 
plt.ylabel("Median Value of Homes (MEDV)") 
plt.show() 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
mse = mean_squared_error(y_test, y_pred) 
rmse = mse ** 0.5 
r2 = r2_score(y_test, y_pred) 
print("MSE:", mse) 
print("RMSE:", rmse) 
print("R² Score:", r2) 
plt.scatter(X_test, y_test, color='blue', label='Actual') 
plt.plot(X_test, y_pred, color='red', label='Predicted') 
plt.title("Regression Line: LSTAT vs MEDV") 
plt.xlabel("Average of lower status of the population ") 
plt.ylabel("Median Value") 
plt.legend() 
plt.show()
