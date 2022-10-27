import pandas as pd

from project.connections import mongo
from project.queuing_theory.utils import load_estimations_data


# Load estimations data
df_queues_list = load_estimations_data(mongo=mongo)
df_queues = pd.DataFrame(data=df_queues_list)

# Define best cluster setup
m_m_1 = df_queues[df_queues["nodes"].isin([1]) & df_queues["cores"].isin([1]) & df_queues["memory"].isin([1])]
m_m_1_fast = df_queues[df_queues["nodes"].isin([1]) & df_queues["cores"].isin([2]) & df_queues["memory"].isin([2])]
m_m_k = df_queues[df_queues["nodes"].isin([4]) & df_queues["cores"].isin([1]) & df_queues["memory"].isin([1])]
m_m_k_fast = df_queues[df_queues["nodes"].isin([4]) & df_queues["cores"].isin([2]) & df_queues["memory"].isin([2])]

queues_e_t_stats = {
    "M/M/1": {
        "mean": m_m_1["response_time"].mean(),
        "std": m_m_1["response_time"].std()
    },
    "M/M/1-fast": {
        "mean": m_m_1_fast["response_time"].mean(),
        "std": m_m_1_fast["response_time"].std()
    },
    "M/M/k": {
        "mean": m_m_k["response_time"].mean(),
        "std": m_m_k["response_time"].std()
    },
    "M/M/k-fast": {
        "mean": m_m_k_fast["response_time"].mean(),
        "std": m_m_k_fast["response_time"].std()
    }
}

comparisons = {}
for setup_a, values_a in queues_e_t_stats.items():
    for setup_b, values_b in queues_e_t_stats.items():
        if setup_a == setup_b or \
                f"{setup_a}_vs_{setup_b}" in comparisons.keys() or \
                f"{setup_b}_vs_{setup_a}" in comparisons.keys():
            continue

        comparisons[f"{setup_a}_vs_{setup_b}"] = {
            "ratio": round(values_a["mean"] / values_b["mean"], 2)
        }

print("------------ COMPARISONS - ESTIMATES --------------")
df_comparisons = pd.DataFrame(data=comparisons)
print(df_comparisons.to_string())
