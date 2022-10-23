import json
import os
from datetime import datetime

from pytz import utc

from project import datasets, networks, CONFIGS_BASE_PATH, RESULTS_BASE_PATH, epochs, learning_rates, DATA_PARALLELISM


# Generate experiments files
from project.connections import mongo
from project.models.configs import Configs
from project.models.results import Results

for key, dataset in datasets.items():
    for network in networks:
        for epoch in epochs:
            for exp_learning_rate, learning_rate in learning_rates.items():
                experiment_configs_path = os.path.join(CONFIGS_BASE_PATH, f"{key}_{network.lower()}", f"epoch_{epoch}",
                                                       f"experiment_{exp_learning_rate}_{epoch}.json")
                experiment_results_path = os.path.join(RESULTS_BASE_PATH, f"{key}_{network.lower()}", f"epoch_{epoch}",
                                                       f"experiment_{exp_learning_rate}_{epoch}.json")

                # Load experiment configurations
                with open(experiment_configs_path) as conf_f:
                    experiment_configs = Configs(**json.load(conf_f))

                # Load experiment results
                with open(experiment_results_path) as res_f:
                    experiment_results = Results(**json.load(res_f))

                if len(experiment_results.accuracy) == 0:
                    continue

                start_time = datetime.strptime(experiment_results.start_time, "%Y-%m-%d %H:%M:%S").timestamp()
                end_time = datetime.strptime(experiment_results.end_time, "%Y-%m-%d %H:%M:%S").timestamp()

                data_to_store = {
                    "nodes": DATA_PARALLELISM,
                    "cores": experiment_configs.trainTasks[0].jobClassParameters[0].systemParameters.configurations
                    ["default"]["cores"],
                    "memory": experiment_configs.trainTasks[0].jobClassParameters[0].systemParameters.configurations
                    ["default"]["memory"],
                    "dataset": dataset,
                    "network": network,
                    "epochs": epoch,
                    "learning_rate": learning_rate,
                    "average_loss": experiment_results.loss,
                    "accuracy": experiment_results.accuracy,
                    "duration_s": end_time - start_time,
                    "created": datetime.now(tz=utc)
                }

                # Add to database
                mongo.replace_one(
                    collection="doe_data",
                    data=data_to_store,
                    query={
                        "dataset": data_to_store["dataset"],
                        "network": data_to_store["network"],
                        "epochs": data_to_store["epochs"],
                        "learning_rate": data_to_store["learning_rate"],
                    }
                )

print("Parser job done!")
