from copy import deepcopy

from project.queuing_theory import N


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

        service_rate = k * (1.0 / e_s)
        N_q = 1
        rho_utilization = (N - N_q) / N
        lambda_arrival_rate = rho_utilization * service_rate
        response_time = (1.0 / lambda_arrival_rate) + e_s

        q_values["response_time"] = response_time

        # M/M/k fast values
        response_time_fast = (1.0 / (2.0 * lambda_arrival_rate)) + e_s / 2.0

        df_m_m_k_fast_dict = deepcopy(q_values)
        df_m_m_k_fast_dict["duration_s"] = round(e_s / 2.0, 2)
        df_m_m_k_fast_dict["response_time"] = round(response_time_fast, 2)
        df_m_m_k_fast_dict["memory"] = "2Gi"
        df_m_m_k_fast_dict["cores"] = "1000m"
        df_m_m_k_fast_list.append(df_m_m_k_fast_dict)

    return df_m_m_k_fast_list


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
        service_rate = 1.0 / e_s
        rho_utilization = N / (1.0 + N)
        lambda_arrival_rate = rho_utilization * service_rate
        response_time = 1.0 / (service_rate - lambda_arrival_rate)

        df_m_m_1_dict = deepcopy(q_values)
        df_m_m_1_dict["duration_s"] = e_s
        df_m_m_1_dict["response_time"] = round(response_time, 2)
        df_m_m_1_dict["nodes"] = 1
        df_m_m_1_list.append(df_m_m_1_dict)

        # M/M/1-fast
        response_time_fast = 1.0 / (2.0 * service_rate - lambda_arrival_rate)

        df_m_m_1_fast_dict = deepcopy(q_values)
        df_m_m_1_fast_dict["duration_s"] = round(e_s / 2.0, 2)
        df_m_m_1_fast_dict["response_time"] = round(response_time_fast, 2)
        df_m_m_1_fast_dict["nodes"] = 1
        df_m_m_1_fast_dict["memory"] = "2Gi"
        df_m_m_1_fast_dict["cores"] = "1000m"
        df_m_m_1_fast_list.append(df_m_m_1_fast_dict)

    return df_m_m_1_list, df_m_m_1_fast_list
