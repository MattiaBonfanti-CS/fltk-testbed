import pandas as pd

from project.connections import mongo
from project.queuing_theory import N
from project.queuing_theory.utils import load_experiments_data

# Load data from the DB
df_m_m_k_list = load_experiments_data(mongo=mongo)

# M/M/1 and M/M/1-fast queue estimations
df_m_m_1_list = []
df_m_m_1_fast_list = []
for q_values in df_m_m_k_list:
    k = q_values["nodes"]
    e_t = k * q_values["response_time"]

    # M/M/1
    lambda_arrival_rate = N / e_t
    service_rate = (1.0 / e_t) + lambda_arrival_rate
    rho_utilization = lambda_arrival_rate / service_rate
    response_time = 1.0 / (service_rate - lambda_arrival_rate)

    df_m_m_1_dict = {
        **q_values,
        "lambda_arrival_rate": lambda_arrival_rate,
        "service_rate": service_rate,
        "rho_utilization": rho_utilization,
        "response_time": response_time,
        "n_jobs": lambda_arrival_rate * (1.0 / (service_rate - lambda_arrival_rate))
    }
    df_m_m_1_dict["nodes"] = 1
    df_m_m_1_list.append(df_m_m_1_dict)

    # M/M/1-fast
    df_m_m_1_fast_dict = {
        **q_values,
        "lambda_arrival_rate": lambda_arrival_rate,
        "service_rate": 2.0 * service_rate,
        "rho_utilization": rho_utilization / 2.0,
        "response_time": 1.0 / (2.0 * service_rate - lambda_arrival_rate),
        "n_jobs": lambda_arrival_rate * (1.0 / (2.0 * service_rate - lambda_arrival_rate))
    }
    df_m_m_1_fast_dict["nodes"] = 1
    df_m_m_1_fast_dict["memory"] = "2Gi"
    df_m_m_1_fast_dict["cores"] = "1000m"

    df_m_m_1_fast_list.append(df_m_m_1_fast_dict)

df_queues_list = df_m_m_1_list + df_m_m_1_fast_list + df_m_m_k_list
df_queues = pd.DataFrame(data=df_queues_list)
print(df_queues)
print()
