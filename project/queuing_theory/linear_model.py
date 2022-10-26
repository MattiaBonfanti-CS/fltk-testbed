import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from project.connections import mongo
from project.queuing_theory.utils import load_estimations_data


# Load estimations data
df_queues_list = load_estimations_data(mongo=mongo)

df_queues = pd.DataFrame(data=df_queues_list)
print(df_queues)
print()

# Prepare data for linear regression to estimate response time
y = df_queues["response_time"]
X = df_queues[["nodes", "cores", "memory", "dataset", "network", "epochs", "learning_rate", "accuracy"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

mse_accuracy = mean_squared_error(y_test, y_pred)
r2_accuracy = r2_score(y_test, y_pred)

print(f"Response time E[T] predictions: MSE = {mse_accuracy}, R2: {r2_accuracy}")
