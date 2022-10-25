ANOVA_FACTORS = "accuracy ~ " \
                "C(dataset) + C(network) + C(epochs) + C(learning_rate) + " \
                "C(dataset):C(network) + C(dataset):C(epochs) + C(dataset):C(learning_rate) + " \
                "C(network):C(epochs) + C(network):C(learning_rate) + " \
                "C(epochs):C(learning_rate) + " \
                "C(dataset):C(network):C(epochs) + C(dataset):C(network):C(learning_rate) + " \
                "C(dataset):C(epochs):C(learning_rate) + " \
                "C(network):C(epochs):C(learning_rate) + " \
                "C(dataset):C(network):C(epochs):C(learning_rate)"
