import json
import os

from project import datasets, networks, CONFIGS_BASE_PATH, RESULTS_BASE_PATH, epochs, learning_rates, DATA_PARALLELISM
from project.models.results import Results
from project.models.templates.configs import CONFIGS_TEMPLATE
from project.models.templates.results import RESULTS_TEMPLATE
from project.models.configs import Configs


# Generate experiments files
for key, dataset in datasets.items():
    for network in networks:
        configs_path = os.path.join(CONFIGS_BASE_PATH, f"{key}_{network.lower()}")
        results_path = os.path.join(RESULTS_BASE_PATH, f"{key}_{network.lower()}")

        for epoch in epochs:
            epoch_configs_path = os.path.join(configs_path, f"epoch_{epoch}")
            epoch_results_path = os.path.join(results_path, f"epoch_{epoch}")

            # Create paths if not exists
            if not os.path.exists(epoch_configs_path):
                os.makedirs(epoch_configs_path)

            if not os.path.exists(epoch_results_path):
                os.makedirs(epoch_results_path)

            for exp_learning_rate, learning_rate in learning_rates.items():
                learning_rate_epoch_configs_path = os.path.join(epoch_configs_path,
                                                                f"experiment_{exp_learning_rate}_{epoch}.json")
                learning_rate_epoch_results_path = os.path.join(epoch_results_path,
                                                                f"experiment_{exp_learning_rate}_{epoch}.json")

                if os.path.exists(learning_rate_epoch_configs_path) \
                        and os.path.exists(learning_rate_epoch_results_path):
                    print("Files already created")
                    continue

                # Load models
                experiment_configs = Configs(**CONFIGS_TEMPLATE)
                experiment_results = Results(**RESULTS_TEMPLATE)

                # Set values for specific experiment
                experiment_configs.trainTasks[0].jobClassParameters[0].networkConfiguration.dataset = dataset
                experiment_configs.trainTasks[0].jobClassParameters[0].networkConfiguration.network = network

                experiment_configs.trainTasks[0].jobClassParameters[0].systemParameters.dataParallelism = \
                    DATA_PARALLELISM

                experiment_configs.trainTasks[0].jobClassParameters[0].hyperParameters.default.totalEpochs = epoch
                experiment_configs.trainTasks[0].jobClassParameters[0].hyperParameters.default.optimizerConfig.\
                    learningRate = learning_rate

                # Create configs file
                with open(learning_rate_epoch_configs_path, "w", encoding="utf-8") as conf_f:
                    json.dump(experiment_configs.dict(by_alias=True), conf_f, ensure_ascii=False, indent=2)
                    print(f"Created: {learning_rate_epoch_configs_path}")

                # Create results file
                with open(learning_rate_epoch_results_path, "w", encoding="utf-8") as res_f:
                    json.dump(experiment_results.dict(), res_f, ensure_ascii=False, indent=2)
                    print(f"Created: {learning_rate_epoch_results_path}")

                print()
