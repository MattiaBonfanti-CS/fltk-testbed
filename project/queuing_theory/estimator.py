from datetime import datetime

from pytz import utc

from project.connections import mongo
from project.queuing_theory.utils import load_experiments_data, estimate_m_m_k_fast_queue, \
    estimate_m_m_1_m_m_1_fast_queues

# Load data from the DB
df_m_m_k_list = load_experiments_data(mongo=mongo)

# M/M/k-fast queue estimations
df_m_m_k_fast_list = estimate_m_m_k_fast_queue(df_m_m_k_list)

# M/M/1 and M/M/1-fast queue estimations
df_m_m_1_list, df_m_m_1_fast_list = estimate_m_m_1_m_m_1_fast_queues(df_m_m_k_list)

# Full queues data list
df_queues_list = df_m_m_1_list + df_m_m_1_fast_list + df_m_m_k_list + df_m_m_k_fast_list

# Store to DB
for result in df_queues_list:
    # Add to database
    result["created"] = datetime.now(tz=utc)

    mongo.replace_one(
        collection="queue_data",
        data=result,
        query={
            "nodes": result["nodes"],
            "memory": result["memory"],
            "cores": result["cores"],
            "dataset": result["dataset"],
            "network": result["network"],
            "epochs": result["epochs"],
            "learning_rate": result["learning_rate"]
        }
    )

print("Estimator job completed!")
