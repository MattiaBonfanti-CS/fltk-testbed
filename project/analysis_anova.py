import warnings

import pandas as pd
import statsmodels.api as sm

from project.connections import mongo
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


# Set constants
CONFIDENCE_LEVEL = 0.05

# Load data from the DB
experiments_results = mongo.find(collection="doe_data", query={})
df_list = []
for result in experiments_results:
    for accuracy in result["accuracy"]:
        df_list.append(
            {
                "dataset": result["dataset"],
                "network": result["network"],
                "epochs": result["epochs"],
                "learning_rate": result["learning_rate"],
                "accuracy": accuracy
            }
        )

# Create data frame
df = pd.DataFrame(data=df_list)
print(df)
print()

# ANOVA Analysis
model = ols("accuracy ~ "
            "C(dataset) + C(network) + C(epochs) + C(learning_rate) + "
            "C(dataset):C(network) + C(dataset):C(epochs) + C(dataset):C(learning_rate) + "
            "C(network):C(epochs) + C(network):C(learning_rate) + "
            "C(epochs):C(learning_rate) + "
            "C(dataset):C(network):C(epochs) + C(dataset):C(network):C(learning_rate) + "
            "C(dataset):C(epochs):C(learning_rate) + "
            "C(network):C(epochs):C(learning_rate) + "
            "C(dataset):C(network):C(epochs):C(learning_rate)", data=df).fit()

# model = ols("accuracy ~ "
#             "C(epochs) + C(learning_rate) + "
#             "C(epochs):C(learning_rate)", data=df).fit()

anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result.to_string())

# Save to DB
anova_dict = anova_result.to_dict()
anova_dict["rejected"] = {}

for parameter, p_value in anova_dict["PR(>F)"].items():
    if parameter != "Residual":
        anova_dict["rejected"][parameter] = p_value > CONFIDENCE_LEVEL

mongo.insert_one(collection="anova_data", data=anova_dict)

print("ANOVA Analysis completed")
