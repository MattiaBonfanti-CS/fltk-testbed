import warnings

import pandas as pd
import statsmodels.api as sm

from project.anova import ANOVA_FACTORS
from project.anova.utils import store_anova_results, load_experiments_data
from project.connections import mongo
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


# Set constants
CONFIDENCE_LEVEL = 0.05

# Load data from the DB
df_2k_factorial_list = load_experiments_data(mongo=mongo, full_factorial=False)

# Create data frame for 2-k factorial analysis
df_2k_factorial = pd.DataFrame(data=df_2k_factorial_list)
print(df_2k_factorial)
print()

# ANOVA Analysis

# 2-k factorial analysis
print("-------------- 2-K FACTORIAL ANOVA ----------------")

anova_2k_factorial_model = ols(ANOVA_FACTORS, data=df_2k_factorial).fit()
print(anova_2k_factorial_model.summary())
print()

anova_2k_factorial_result = sm.stats.anova_lm(anova_2k_factorial_model, typ=2)
print(anova_2k_factorial_result.to_string())
print()

# Save to DB
store_anova_results(
    anova_results=anova_2k_factorial_result,
    anova_type="2-k factorial",
    confidence_level=CONFIDENCE_LEVEL,
    mongo=mongo
)

print("2-k ANOVA Analysis completed")
