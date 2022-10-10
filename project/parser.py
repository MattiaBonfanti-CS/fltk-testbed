import json
import os
from datetime import datetime

import pandas as pd
from pytz import utc

directory = "../logging/cloud_project_experiment"
filename = "federator.csv"
experiment_id = 1
experiments_data = []
for directory_name in os.listdir(directory):
    # Load experiment parameters
    with open(f"configs/project_arrival_config_comb_{experiment_id}.json") as f:
        experiment_json = json.load(f)

    # Load experiment results
    file_path = os.path.join(directory, f"{directory_name}/{filename}")
    experiment_data = pd.read_csv(file_path, header=0)

    data_to_store = {
        "k_clients": experiment_json["trainTasks"][0]["jobClassParameters"][0]["learningParameters"]["clientsPerRound"],
        "learning_rate": experiment_json["trainTasks"][0]["jobClassParameters"][0]["hyperParameters"]["configurations"]["Worker"]["optimizerConfig"]["learningRate"],
        "epochs": experiment_json["trainTasks"][0]["jobClassParameters"][0]["learningParameters"]["epochsPerRound"],
        "repetitions": [],
        "created": datetime.now(tz=utc)
    }

    for index, row in experiment_data.iterrows():
        data_to_store["repetitions"].append(
            {
                "repetition_id": row["round_id"] + 1,
                "total_duration": row["round_duration"],
                "train_duration": row["round_duration"] - row["test_duration"],
                "test_duration": row["test_duration"],
                "send_receive_duration": row["send_receive_duration"],
                "test_loss": row["test_loss"],
                "test_accuracy": row["test_accuracy"],
                "executed": datetime.utcfromtimestamp(row["timestamp"])
            }
        )

    experiments_data.append(data_to_store)
    experiment_id += 1

print(experiments_data)
