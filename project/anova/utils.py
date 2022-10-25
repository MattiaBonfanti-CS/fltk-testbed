from project import learning_rates, epochs


def load_experiments_data(mongo, full_factorial=True):
    """
    Load the experiments results.

    :param mongo: The DB connection.
    :param full_factorial: The full_factorial flag.
    :return: The list of data from the DB.
    """
    learning_rate_range = (min(learning_rates.values()), max(learning_rates.values()))
    epochs_range = (min(epochs), max(epochs))

    experiments_results = mongo.find(collection="doe_data", query={})
    df_list = []
    for result in experiments_results:
        for accuracy in result["accuracy"]:
            # Only use max and min values for 2-k factorial analysis
            if full_factorial or (result["learning_rate"] in learning_rate_range and result["epochs"] in epochs_range):
                df_list.append(
                    {
                        "dataset": result["dataset"],
                        "network": result["network"],
                        "epochs": result["epochs"],
                        "learning_rate": result["learning_rate"],
                        "accuracy": accuracy
                    }
                )

    return df_list


def store_anova_results(anova_results, anova_type, confidence_level, mongo):
    """
    Store the ANOVA analysis results to the mongo database.

    :param anova_results: The ANOVA results.
    :param anova_type: The ANOVA analysis type.
    :param confidence_level: The confidence level.
    :param mongo: The DB connection.
    """
    anova_dict = anova_results.to_dict()
    anova_dict["type"] = anova_type
    anova_dict["rejected"] = {}

    for parameter, p_value in anova_dict["PR(>F)"].items():
        if parameter != "Residual":
            anova_dict["rejected"][parameter] = p_value > confidence_level

    mongo.replace_one(collection="anova_data", data=anova_dict, query={"type": anova_type})
