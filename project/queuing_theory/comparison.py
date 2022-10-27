import json
import os
import warnings
from datetime import datetime

import pandas as pd

from project import queues, RESULTS_QUEUE_BASE_PATH
from project.connections import mongo
from project.models.results import ResultsQueue
from project.queuing_theory import N
from project.queuing_theory.utils import load_estimations_data, compare_queues

warnings.filterwarnings("ignore")

# Load estimations data
df_queues_list = load_estimations_data(mongo=mongo)
df_queues = pd.DataFrame(data=df_queues_list)

# Define best cluster setup
m_m_1 = df_queues[df_queues["nodes"].isin([1]) & df_queues["cores"].isin([1]) & df_queues["memory"].isin([1])]
m_m_1_fast = df_queues[df_queues["nodes"].isin([1]) & df_queues["cores"].isin([2]) & df_queues["memory"].isin([2])]
m_m_k = df_queues[df_queues["nodes"].isin([4]) & df_queues["cores"].isin([1]) & df_queues["memory"].isin([1])]
m_m_k_fast = df_queues[df_queues["nodes"].isin([4]) & df_queues["cores"].isin([2]) & df_queues["memory"].isin([2])]

comparisons = compare_queues(m_m_1, m_m_1_fast, m_m_k, m_m_k_fast)

print("------------ COMPARISONS - ALL ESTIMATES --------------")
df_comparisons = pd.DataFrame(data=comparisons)
print(df_comparisons.to_string())
print()

# Compare estimates with 0.1 learning rate with executed tests
m_m_1_lr_0_1 = m_m_1[df_queues["learning_rate"].isin([0.1])]
m_m_1_fast_lr_0_1 = m_m_1_fast[df_queues["learning_rate"].isin([0.1])]
m_m_k_lr_0_1 = m_m_k[df_queues["learning_rate"].isin([0.1])]
m_m_k_fast_lr_0_1 = m_m_k_fast[df_queues["learning_rate"].isin([0.1])]

# Read queue experiments results
df_queues_experiments_list = []
for key, queue in queues.items():
    experiment_results_path = os.path.join(RESULTS_QUEUE_BASE_PATH, f"experiment_{queue}.json")
    # Load experiment results
    with open(experiment_results_path) as res_f:
        experiment_results = ResultsQueue(**json.load(res_f))

    for i, epoch in enumerate(experiment_results.epochs):
        start_time = datetime.strptime(experiment_results.start_time[i], "%Y-%m-%d %H:%M:%S").timestamp()
        end_time = datetime.strptime(experiment_results.end_time[i], "%Y-%m-%d %H:%M:%S").timestamp()

        service_rate = 1.0 / (end_time - start_time)

        if queue == "m_m_1" or queue == "m_m_1_fast":
            rho_utilization = N / (1.0 + N)
            lambda_arrival_rate = rho_utilization * service_rate
            response_time = 1.0 / (service_rate - lambda_arrival_rate)
        else:
            N_q = 1
            rho_utilization = (N - N_q) / N
            lambda_arrival_rate = rho_utilization * service_rate
            response_time = (1.0 / lambda_arrival_rate) + (end_time - start_time)

        df_queues_experiments_list.append({
            "duration_s": end_time - start_time,
            "response_time": response_time,
            "epochs": epoch,
            "learning_rate": experiment_results.learning_rate,
            "nodes": 1 if queue == "m_m_1" or queue == "m_m_1_fast" else 4,
            "cores": 1 if queue == "m_m_1" or queue == "m_m_k" else 2,
            "memory": 1 if queue == "m_m_1" or queue == "m_m_k" else 2
        })

df_queues_experiments = pd.DataFrame(data=df_queues_experiments_list)

m_m_1_exp = df_queues_experiments[
    df_queues_experiments["nodes"].isin([1]) &
    df_queues_experiments["cores"].isin([1]) &
    df_queues_experiments["memory"].isin([1])]
m_m_1_fast_exp = df_queues_experiments[
    df_queues_experiments["nodes"].isin([1]) &
    df_queues_experiments["cores"].isin([2]) &
    df_queues_experiments["memory"].isin([2])]
m_m_k_exp = df_queues_experiments[
    df_queues_experiments["nodes"].isin([4]) &
    df_queues_experiments["cores"].isin([1]) &
    df_queues_experiments["memory"].isin([1])]
m_m_k_fast_exp = df_queues_experiments[
    df_queues_experiments["nodes"].isin([4]) &
    df_queues_experiments["cores"].isin([2]) &
    df_queues_experiments["memory"].isin([2])]

comparisons_exp = compare_queues(m_m_1_exp, m_m_1_fast_exp, m_m_k_exp, m_m_k_fast_exp)

print("------------ COMPARISONS - EXPERIMENTS --------------")
df_comparisons_exp = pd.DataFrame(data=comparisons_exp)
print(df_comparisons_exp.to_string())
print()
