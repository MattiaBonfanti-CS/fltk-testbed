from copy import deepcopy
from math import factorial

from project.queuing_theory import USED_PODS_M_M_1, AVAILABLE_NODES, USED_PODS_M_M_K


def load_experiments_data(mongo):
    """
    Load the experiments results.

    :param mongo: The DB connection.
    :return: The list of data from the DB.
    """
    experiments_results = mongo.find(collection="doe_data", query={})
    df_list = []
    for result in experiments_results:
            df_list.append(
                {
                    "nodes": result["nodes"],
                    "cores": result["cores"],
                    "memory": result["memory"],
                    "dataset": result["dataset"],
                    "network": result["network"],
                    "epochs": result["epochs"],
                    "learning_rate": result["learning_rate"],
                    "average_loss": result["average_loss"],
                    "accuracy": result["accuracy"],
                    "duration_s": result["duration_s"]
                }
            )

    return df_list


def load_estimations_data(mongo):
    """
    Load the estimations results.

    :param mongo: The DB connection.
    :return: The list of data from the DB.
    """
    experiments_results = mongo.find(collection="queue_data", query={})
    df_list = []
    for result in experiments_results:
        for accuracy in result["accuracy"]:
            df_list.append(
                {
                    "nodes": result["nodes"],
                    "cores": 1 if result["cores"] == "500m" else 2,
                    "memory": 1 if result["memory"] == "1Gi" else 2,
                    "dataset": 1 if result["dataset"] == "mnist" else -1,
                    "network": 1 if result["network"] == "FashionMNISTCNN" else -1,
                    "epochs": result["epochs"],
                    "learning_rate": result["learning_rate"],
                    "accuracy": accuracy,
                    "response_time": result["response_time"]
                }
            )

    return df_list


def response_time_m_m_k(k, e_s):
    """
    Calculate response time for M/M/K type queues

    :param k: The available servers.
    :param e_s: The service time for one server.

    :return: The overall response time.
    """
    service_rate = k * (1.0 / e_s)
    rho_utilization = USED_PODS_M_M_K / AVAILABLE_NODES
    lambda_arrival_rate = rho_utilization * service_rate

    # Calculate pi_0
    sum_factor = 0
    for i in range(k):
        sum_factor += pow(k * rho_utilization, i) / factorial(i)
    constant_factor = pow(k * rho_utilization, k) / (factorial(k) * (1.0 - rho_utilization))
    pi_0 = 1.0 / (sum_factor + constant_factor)

    P_q = constant_factor * pi_0
    N_q = P_q * rho_utilization / (1.0 - rho_utilization)
    e_t_q = N_q / lambda_arrival_rate

    return e_t_q + (1.0 / service_rate)


def estimate_m_m_k_fast_queue(m_m_k_data):
    """
    Estimate M/M/k-fast queue.
    
    :param m_m_k_data: M/M/k queue data.
    :return: The M/M/k-fast queue estimations.
    """
    df_m_m_k_fast_list = []
    for q_values in m_m_k_data:
        # M/M/k values
        k = q_values["nodes"]
        e_s = q_values["duration_s"]

        response_time = response_time_m_m_k(k, e_s)
        q_values["response_time"] = round(response_time, 2)

        # M/M/k fast values
        response_time_fast = response_time_m_m_k(k, e_s / 2.0)

        df_m_m_k_fast_dict = deepcopy(q_values)
        df_m_m_k_fast_dict["duration_s"] = round(e_s / 2.0, 2)
        df_m_m_k_fast_dict["response_time"] = round(response_time_fast, 2)
        df_m_m_k_fast_dict["memory"] = "2Gi"
        df_m_m_k_fast_dict["cores"] = "1000m"
        df_m_m_k_fast_list.append(df_m_m_k_fast_dict)

    return df_m_m_k_fast_list


def response_time_m_m_1(e_s):
    """
    Calculate response time for M/M/1 type queues

    :param e_s: The service time for one server.

    :return: The overall response time.
    """
    service_rate = 1.0 / e_s
    rho_utilization = USED_PODS_M_M_1 / AVAILABLE_NODES
    lambda_arrival_rate = rho_utilization * service_rate

    return 1.0 / (service_rate - lambda_arrival_rate)


def estimate_m_m_1_m_m_1_fast_queues(m_m_k_data):
    """
    Estimate M/M/1 and M/M/1-fast queues.

    :param m_m_k_data: M/M/k queue data.
    :return: The M/M/1 and M/M/1-fast queues estimations.
    """
    df_m_m_1_list = []
    df_m_m_1_fast_list = []

    for q_values in m_m_k_data:
        k = q_values["nodes"]
        e_s = k * q_values["duration_s"]

        # M/M/1
        response_time = response_time_m_m_1(e_s)

        df_m_m_1_dict = deepcopy(q_values)
        df_m_m_1_dict["duration_s"] = e_s
        df_m_m_1_dict["response_time"] = round(response_time, 2)
        df_m_m_1_dict["nodes"] = 1
        df_m_m_1_list.append(df_m_m_1_dict)

        # M/M/1-fast
        response_time_fast = response_time_m_m_1(e_s / 2.0)

        df_m_m_1_fast_dict = deepcopy(q_values)
        df_m_m_1_fast_dict["duration_s"] = round(e_s / 2.0, 2)
        df_m_m_1_fast_dict["response_time"] = round(response_time_fast, 2)
        df_m_m_1_fast_dict["nodes"] = 1
        df_m_m_1_fast_dict["memory"] = "2Gi"
        df_m_m_1_fast_dict["cores"] = "1000m"
        df_m_m_1_fast_list.append(df_m_m_1_fast_dict)

    return df_m_m_1_list, df_m_m_1_fast_list


def compare_queues(m_m_1, m_m_1_fast, m_m_k, m_m_k_fast):
    """
    Compare queues.
    """
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

    return comparisons
