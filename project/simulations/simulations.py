import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression

AVAILABLE_DATASETS = list({"mnist", "fashion-mnist"})
AVAILABLE_NETWORKS = list({"MNISTCNN", "FashionMNISTCNN"})
AVAILABLE_LEARNING_RATES = list({1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6})

GENERATOR = default_rng()

class Job():
    def __init__(self, dataset: str, network: str, learning_rate: float):
        self._dataset = dataset
        self._network = network
        self._learning_rate = learning_rate

    def __eq__(self, other):
        if isinstance(other, Job):
            return self._dataset == other._dataset and self._network == other._network and math.isclose(self._learning_rate, other._learning_rate)
        return False

    def __hash__(self):
        return hash((self._dataset, self._network, self._learning_rate))

    def __str__(self):
        return "Job(" + self._dataset + ", " + self._network + ", " + str(self._learning_rate) + ")"

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = network

    @property
    def learning_rate(self):
        return self.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

def create_model(data_list):
    xs = pd.DataFrame(data=data_list)

    ys = xs["accuracy"]
    xs.drop("accuracy", inplace=True, axis=1)

    xs["dataset"] = xs["dataset"].apply(
        lambda dataset: 1 if dataset == "mnist" else -1
    )
    xs["network"] = xs["network"].apply(
        lambda network: 1 if network == "FashionMNISTCNN" else -1
    )

    return LinearRegression().fit(xs, ys)

def run_simulation(
        num_jobs: int,
        arrival_rate_low: float,
        arrival_rate_high: float,
        num_epochs_low: int,
        num_epochs_high: int,
        avg_time_per_epoch: dict[Job, float],
        model):
    def helper(
            previous_arrival_time: float,
            previous_end_time: float,
            inter_arrival_time: float,
            service_time: float):
        arrival_time = previous_arrival_time + inter_arrival_time
        start_time = max(arrival_time, previous_end_time)
        end_time = start_time + service_time
        response_time = end_time - arrival_time

        return response_time, arrival_time, end_time

    num_epochs_avg = (num_epochs_low + num_epochs_high) // 2

    results = np.empty(num_jobs)

    previous_arrival_time = 0
    previous_end_time = 0

    previous_arrival_time_improved = 0
    previous_end_time_improved = 0

    is_low_arrival_rate = False
    interval = -1

    for i in range(num_jobs):
        if interval <= 0:
            interval = GENERATOR.integers(100, 500, endpoint=True)
            is_low_arrival_rate = not is_low_arrival_rate

        interval -= 1

        arrival_rate = arrival_rate_low if is_low_arrival_rate else arrival_rate_high
        inter_arrival_time = np.random.exponential(1 / arrival_rate)

        num_epochs = num_epochs_avg
        num_epochs_improved = num_epochs_high if is_low_arrival_rate else num_epochs_low

        # Generate a job
        dataset = random.choice(AVAILABLE_DATASETS)
        network = random.choice(AVAILABLE_NETWORKS)
        learning_rate = random.choice(AVAILABLE_LEARNING_RATES)

        dataset_formatted = 1 if dataset == "mnist" else -1
        network_formatted = 1 if network == "FashionMNISTCNN" else -1

        job = Job(dataset, network, learning_rate)

        accuracy = model.predict(pd.DataFrame.from_dict({"dataset" : [dataset_formatted], "network" : [network_formatted], "epochs" : [num_epochs], "learning_rate" : [learning_rate]}))
        accuracy_improved = model.predict(pd.DataFrame.from_dict({"dataset" : [dataset_formatted], "network" : [network_formatted], "epochs" : [num_epochs_improved], "learning_rate" : [learning_rate]}))

        # Get the result for the generated job without the improvement
        response_time, arrival_time, end_time = helper(
            previous_arrival_time,
            previous_end_time,
            inter_arrival_time,
            avg_time_per_epoch[job] * num_epochs)

        result = accuracy / response_time

        previous_arrival_time = arrival_time
        previous_end_time = end_time

        # Get the result for the generated job with the improvement
        response_time_improved, arrival_time_improved, end_time_improved = helper(
            previous_arrival_time_improved,
            previous_end_time_improved,
            inter_arrival_time,
            avg_time_per_epoch[job] * num_epochs_improved)

        result_improved = accuracy_improved / response_time_improved

        previous_arrival_time_improved = arrival_time_improved
        previous_end_time_improved = end_time_improved

        # Compute the improvement in the result
        improvement = result_improved - result
        results[i] = (improvement / result) * 100

    # Return the average improvement
    return results

def run_experiment(
        num_jobs: int,
        utilisation_low: float,
        utilisation_high: float,
        num_epochs_low: int,
        num_epochs_high: int):
    # Get the average service time for every combintaion of hyperparameters from the data
    epochs_per_job = {}
    avg_time_per_epoch_per_job = {}

    for dataset in AVAILABLE_DATASETS:
        for network in AVAILABLE_NETWORKS:
            for learning_rate in AVAILABLE_LEARNING_RATES:
                job = Job(dataset, network, learning_rate)
                epochs_per_job[job] = 0
                avg_time_per_epoch_per_job[job] = 0

    with open("jobs.json", "r") as f:
        jobs = json.load(f)

    jobs = list(filter(lambda job: job["duration_s"] >= 0, jobs))
    model_data = []

    for job in jobs:
        for accuracy in job["accuracy"]:
                model_data.append(
                    {
                        "dataset": job["dataset"],
                        "network": job["network"],
                        "epochs": job["epochs"],
                        "learning_rate": job["learning_rate"],
                        "accuracy": accuracy
                    }
                )

    model = create_model(model_data)

    for job in jobs:
        dataset = job["dataset"]
        network = job["network"]
        learning_rate = float(job["learning_rate"])
        epochs = int(job["epochs"])
        duration = float(job["duration_s"])

        job = Job(dataset, network, learning_rate)
        epochs_per_job[job] += epochs
        avg_time_per_epoch_per_job[job] += duration

    for job in avg_time_per_epoch_per_job:
        avg_time_per_epoch_per_job[job] /= epochs_per_job[job]

    # Get the average service time from the data
    service_time_avg = np.mean([float(job["duration_s"]) for job in jobs])

    # Get the average service rate
    service_rate_avg = 1 / service_time_avg

    # Get the boundaries of the arrival rate
    arrival_rate_low = utilisation_low * service_rate_avg
    arrival_rate_high = utilisation_high * service_rate_avg

    results = run_simulation(
        num_jobs,
        arrival_rate_low,
        arrival_rate_high,
        num_epochs_low,
        num_epochs_high,
        avg_time_per_epoch_per_job,
        model)

    return [np.mean(results[ : i + 1]) for i in range(num_jobs)]

def generate_statistics(
        num_jobs: int,
        utilisation_low: float,
        utilisation_high: float,
        num_epochs_low: int,
        num_epochs_high: int,
        num_repetitions: int):
    repetitions = []

    for _ in range(num_repetitions):
        repetitions.append(run_experiment(num_jobs, utilisation_low, utilisation_high, num_epochs_low, num_epochs_high))

    result = [np.mean([repetition[i] for repetition in repetitions]) for i in range(num_jobs)]    

    plt.figure()
    plt.plot([i for i in range(1, num_jobs + 1)], result)
    plt.show()

    means = [repetition[-1] for repetition in repetitions]

    print("Parameters:", num_jobs, utilisation_low, utilisation_high, num_epochs_low, num_epochs_high, num_repetitions)
    print("Mean:", np.mean(means))
    print("Variance:", np.var(means))
    print("Standard deviation:", np.std(means))

def main():
    generate_statistics(100_000, 0.25, 0.85, 10, 25, 100)

main()
