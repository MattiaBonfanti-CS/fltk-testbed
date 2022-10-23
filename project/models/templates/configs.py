CONFIGS_TEMPLATE = {
  "trainTasks": [
    {
      "type": "distributed",
      "lambda": 1.5,
      "preemptJobs": False,
      "jobClassParameters": [
        {
          "classProbability": 0.1,
          "priorities": [
            {
              "priority": 1,
              "probability": 0.9
            },
            {
              "priority": 0,
              "probability": 0.1
            }
          ],
          "networkConfiguration": {
            "network": "FashionMNISTCNN",
            "lossFunction": "CrossEntropyLoss",
            "dataset": "mnist"
          },
          "systemParameters": {
            "dataParallelism": 4,
            "configurations": {
              "default": {
                "cores": "500m",
                "memory": "1Gi"
              }
            }
          },
          "hyperParameters": {
            "default": {
              "totalEpochs": 1,
              "batchSize": 128,
              "testBatchSize": 128,
              "learningRateDecay": 0.0002,
              "optimizerConfig": {
                "type": "SGD",
                "learningRate": 0.1,
                "momentum": 0.1
              },
              "schedulerConfig": {
                "schedulerStepSize": 50,
                "schedulerGamma": 0.5,
                "minimumLearningRate": 1e-10
              }
            },
            "configurations": {
              "Master": None,
              "Worker": None
            }
          },
          "learningParameters": {
            "cuda": False
          }
        }
      ]
    }
  ]
}