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
