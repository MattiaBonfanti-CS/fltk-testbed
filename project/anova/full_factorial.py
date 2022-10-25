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
df_full_factorial_list = load_experiments_data(mongo=mongo, full_factorial=True)

# Create data frame for full factorial analysis
df_full_factorial = pd.DataFrame(data=df_full_factorial_list)
print(df_full_factorial)
print()

# ANOVA Analysis

# Full factorial analysis
print("-------------- FULL FACTORIAL ANOVA ----------------")

anova_full_factorial_model = ols(ANOVA_FACTORS, data=df_full_factorial).fit()
print(anova_full_factorial_model.summary())
print()

anova_full_factorial_result = sm.stats.anova_lm(anova_full_factorial_model, typ=2)
print(anova_full_factorial_result.to_string())
print()

# Save to DB
store_anova_results(
    anova_results=anova_full_factorial_result,
    anova_type="full factorial",
    confidence_level=CONFIDENCE_LEVEL,
    mongo=mongo
)

print("Full ANOVA Analysis completed")
