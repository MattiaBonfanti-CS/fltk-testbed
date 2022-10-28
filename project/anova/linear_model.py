import pandas as pd
import seaborn as sns
from pint.testsuite.test_matplotlib import plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

from project.anova import ANOVA_FACTORS
from project.anova.utils import load_experiments_data
from project.connections import mongo


# Load data from the DB
df_linear_model_list = load_experiments_data(mongo=mongo, full_factorial=True)

# Create data frame for full factorial analysis
df_linear_model = pd.DataFrame(data=df_linear_model_list)
print(df_linear_model)
print()

# Linear model ANOVA

print("-------------- LINEAR MODEL ANOVA ----------------")
labels_accuracy = df_linear_model["accuracy"]
train_df_linear_model = df_linear_model
train_df_linear_model["dataset"] = train_df_linear_model["dataset"].apply(
    lambda dataset: 1 if dataset == "mnist" else -1
)
train_df_linear_model["network"] = train_df_linear_model["network"].apply(
    lambda network: 1 if network == "FashionMNISTCNN" else -1
)

X_train, X_test, y_train_accuracy, y_test_accuracy = train_test_split(
    train_df_linear_model, labels_accuracy, test_size=0.2, random_state=42
)

anova_linear_model = ols(ANOVA_FACTORS, data=X_train).fit()
y_pred_accuracy_anova = anova_linear_model.predict(X_test)

mse_accuracy = mean_squared_error(y_test_accuracy, y_pred_accuracy_anova)
r2_accuracy = r2_score(y_test_accuracy, y_pred_accuracy_anova)

print(f"Accuracy predictions: MSE = {mse_accuracy}, R2: {r2_accuracy}")

# Plot results
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 20))

sns.lineplot(x="learning_rate", y="accuracy", data=df_linear_model, ax=ax1)
sns.lineplot(x="epochs", y="accuracy", data=df_linear_model, ax=ax2)

sns.barplot(x="dataset", y="accuracy", data=df_linear_model, ax=ax3)
sns.barplot(x="network", y="accuracy", data=df_linear_model, ax=ax4)

plt.savefig("../plots/anova_linear_sns.png")

f, (ax1, ax2) = plt.subplots(2, figsize=(15, 10))

ax1.scatter(X_test.learning_rate, y_test_accuracy, label="True labels")
ax1.scatter(X_test.learning_rate, y_pred_accuracy_anova, label="Predicted values")

ax1.set_xlabel("Learning rate")
ax1.set_ylabel("Accuracy")

ax2.scatter(X_test.epochs, y_test_accuracy, label="True labels")
ax2.scatter(X_test.epochs, y_pred_accuracy_anova, label="Predicted values")

ax2.set_xlabel("Number of epochs")
ax2.set_ylabel("Accuracy")

ax1.legend()
ax2.legend()

plt.savefig("../plots/anova_linear.png")
