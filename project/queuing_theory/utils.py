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
                    "dataset": result["dataset"],
                    "network": result["network"],
                    "epochs": result["epochs"],
                    "learning_rate": result["learning_rate"],
                    "memory": result["memory"],
                    "cores": result["cores"],
                    "response_time": result["duration_s"]
                }
            )

    return df_list
