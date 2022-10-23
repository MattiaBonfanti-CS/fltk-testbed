from pydantic import BaseModel, Field
from typing import List


class Priority(BaseModel):
    """
    priority class.
    """
    priority: int
    probability: float


class NetworkConfiguration(BaseModel):
    """
    networkConfiguration class.
    """
    network: str = "FashionMNISTCNN"
    lossFunction: str = "CrossEntropyLoss"
    dataset: str = "mnist"


class SystemParameters(BaseModel):
    """
    systemParameters class.
    """
    dataParallelism: int = 2
    configurations: dict = {
        "default": {
            "cores": "500m",
            "memory": "1Gi"
        }
    }


class OptimizerConfig(BaseModel):
    """
    optimizerConfig class.
    """
    type: str = "SGD"
    learningRate: float = 0.1
    momentum: float = 0.1


class DefaultHyperParameters(BaseModel):
    """
    default HyperParameters class.
    """
    totalEpochs: int = 1
    batchSize: int = 128
    testBatchSize: int = 128
    learningRateDecay: float = 0.0002
    optimizerConfig: OptimizerConfig
    schedulerConfig: dict = {
        "schedulerStepSize": 50,
        "schedulerGamma": 0.5,
        "minimumLearningRate": 1e-10
    }


class HyperParameters(BaseModel):
    """
    hyperParameters class.
    """
    default: DefaultHyperParameters
    configurations: dict = {
        "Master": None,
        "Worker": None
    }


class JobClassParameters(BaseModel):
    """
    jobClassParameters class.
    """
    classProbability: float = 0.1
    priorities: List[Priority]
    networkConfiguration: NetworkConfiguration
    systemParameters: SystemParameters
    hyperParameters: HyperParameters
    learningParameters: dict = {
        "cuda": False
    }


class TrainTasks(BaseModel):
    """
    trainTasks class.
    """
    type: str = "distributed"
    lambda_value: float = Field(1.5, alias='lambda')
    preemptJobs: bool = False
    jobClassParameters: List[JobClassParameters]


class Configs(BaseModel):
    """
    Experiments configurations model.
    """
    trainTasks: List[TrainTasks]
