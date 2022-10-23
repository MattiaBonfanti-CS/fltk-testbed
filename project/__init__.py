DATA_PARALLELISM = 4
EXPERIMENT_TYPE = "distributed"
CONFIGS_BASE_PATH = f"./configs/{EXPERIMENT_TYPE}/"
RESULTS_BASE_PATH = f"./results/{EXPERIMENT_TYPE}/"

learning_rates = {
    "10e-1": 0.1,
    "10e-2": 0.01,
    "10e-3": 0.001,
    "10e-4": 0.0001,
    "10e-5": 0.00001,
    "10e-6": 0.000001
}
epochs = [1, 10, 25]
datasets = {"mnist": "mnist", "fashionmnist": "fashion-mnist"}
networks = ["FashionMNISTCNN", "FashionMNISTResNet"]
