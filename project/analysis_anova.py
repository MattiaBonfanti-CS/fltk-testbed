import warnings

import pandas as pd
import statsmodels.api as sm

from project import learning_rates, epochs
from project.connections import mongo
from project.utils import Utils
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


# Set constants
CONFIDENCE_LEVEL = 0.05
LEARNING_RATE_RANGE = (min(learning_rates.values()), max(learning_rates.values()))
EPOCHS_RANGE = (min(epochs), max(epochs))

# Load data from the DB
experiments_results = mongo.find(collection="doe_data", query={})
df_full_factorial_list = []
df_2k_factorial_list = []
for result in experiments_results:
    for accuracy in result["accuracy"]:
        df_full_factorial_list.append(
            {
                "dataset": result["dataset"],
                "network": result["network"],
                "epochs": result["epochs"],
                "learning_rate": result["learning_rate"],
                "accuracy": accuracy
            }
        )

        # Only use max and min values for 2-k factorial analysis
        if result["learning_rate"] in LEARNING_RATE_RANGE and result["epochs"] in EPOCHS_RANGE:
            df_2k_factorial_list.append(
                {
                    "dataset": result["dataset"],
                    "network": result["network"],
                    "epochs": result["epochs"],
                    "learning_rate": result["learning_rate"],
                    "accuracy": accuracy
                }
            )

# Create data frame for full factorial analysis
df_full_factorial = pd.DataFrame(data=df_full_factorial_list)
print(df_full_factorial)
print()

# Create data frame for 2-k factorial analysis
df_2k_factorial = pd.DataFrame(data=df_2k_factorial_list)
print(df_2k_factorial)
print()

# ANOVA Analysis
anova_factors = "accuracy ~ " \
                "C(dataset) + C(network) + C(epochs) + C(learning_rate) + " \
                "C(dataset):C(network) + C(dataset):C(epochs) + C(dataset):C(learning_rate) + " \
                "C(network):C(epochs) + C(network):C(learning_rate) + " \
                "C(epochs):C(learning_rate) + " \
                "C(dataset):C(network):C(epochs) + C(dataset):C(network):C(learning_rate) + " \
                "C(dataset):C(epochs):C(learning_rate) + " \
                "C(network):C(epochs):C(learning_rate) + " \
                "C(dataset):C(network):C(epochs):C(learning_rate)"

# anova_factors = "accuracy ~  " \
#                 "C(epochs) + " \
#                 "C(learning_rate) +  " \
#                 "C(epochs):C(learning_rate)"

# anova_factors = "accuracy ~ " \
#                 "dataset + network + epochs + learning_rate + " \
#                 "dataset:network + dataset:epochs + dataset:learning_rate + " \
#                 "network:epochs + network:learning_rate + " \
#                 "epochs:learning_rate + " \
#                 "dataset:network:epochs + dataset:network:learning_rate + " \
#                 "dataset:epochs:learning_rate + " \
#                 "network:epochs:learning_rate + " \
#                 "dataset:network:epochs:learning_rate"

# Full factorial analysis
print("-------------- FULL FACTORIAL ANOVA ----------------")

anova_full_factorial_model = ols(anova_factors, data=df_full_factorial).fit()
print(anova_full_factorial_model.summary())
print()

anova_full_factorial_result = sm.stats.anova_lm(anova_full_factorial_model, typ=2)
print(anova_full_factorial_result.to_string())
print()

# Save to DB
Utils.store_anova_results(
    anova_results=anova_full_factorial_result,
    anova_type="full factorial",
    confidence_level=CONFIDENCE_LEVEL,
    mongo=mongo
)

# Full factorial analysis
print("-------------- 2-K FACTORIAL ANOVA ----------------")

anova_2k_factorial_model = ols(anova_factors, data=df_2k_factorial).fit()
print(anova_2k_factorial_model.summary())
print()

anova_2k_factorial_result = sm.stats.anova_lm(anova_2k_factorial_model, typ=2)
print(anova_2k_factorial_result.to_string())
print()

# Save to DB
Utils.store_anova_results(
    anova_results=anova_2k_factorial_result,
    anova_type="2-k factorial",
    confidence_level=CONFIDENCE_LEVEL,
    mongo=mongo
)

print("ANOVA Analysis completed")
