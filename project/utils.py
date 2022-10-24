import configparser
import os


class Utils:
    """
    Utility class for helper functions.
    """

    @staticmethod
    def parse_connections_file(key='', conn_cfg_file='connections/connections.cfg'):
        """
        Parse the connections file to retrieve the necessary information.

        :param key: The key to fetch from the file.
        :param conn_cfg_file: The file with the connections details.
        :return: The parsed object from the file.
        """
        conf_parser = configparser.ConfigParser()
        conf_parser.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), conn_cfg_file))

        connections_info = {}
        try:
            connections_info = conf_parser[key]
        except KeyError as error:
            print("KeyError in the {0} file, missing key {1}: {2}".format(conn_cfg_file, key, error))

        return connections_info

    @staticmethod
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

        mongo.insert_one(collection="anova_data", data=anova_dict)
