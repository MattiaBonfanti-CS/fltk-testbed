import json
import os
from datetime import datetime

import pandas as pd
from pytz import utc

from project.connections import mongo

directory = "../logging/cloud_project_experiment"
filename = "federator.csv"
experiment_id = 1
for directory_name in os.listdir(directory):
    # Load experiment parameters
    with open(f"configs/project_arrival_config_comb_{experiment_id}.json") as f:
        experiment_json = json.load(f)

    # Load experiment results
    file_path = os.path.join(directory, f"{directory_name}/{filename}")
    experiment_data = pd.read_csv(file_path, header=0)

    data_to_store = {
        "k_clients": experiment_json["trainTasks"][0]["jobClassParameters"][0]["learningParameters"]["clientsPerRound"],
        "client_cores": experiment_json["trainTasks"][0]["jobClassParameters"][0]["systemParameters"]["configurations"]
        ["Worker"]["cores"],
        "client_memory": experiment_json["trainTasks"][0]["jobClassParameters"][0]["systemParameters"]["configurations"]
        ["Worker"]["memory"],
        "learning_rate": experiment_json["trainTasks"][0]["jobClassParameters"][0]["hyperParameters"]["configurations"]
        ["Worker"]["optimizerConfig"]["learningRate"],
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

    experiment_id += 1

    # Add to database
    mongo.replace_one(
        collection="doe_data",
        data=data_to_store,
        query={
            "learning_rate": data_to_store["learning_rate"],
            "epochs": data_to_store["epochs"]
        }
    )

print("Parser job done!")
