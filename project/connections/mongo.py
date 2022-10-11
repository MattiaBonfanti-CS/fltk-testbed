import os
from urllib import parse

import pymongo
from pymongo import errors as pe

from project.utils import Utils


class Mongo:
    """
    Connection class to the MongoDB.
    """

    def __init__(self, conf_file='connections/connections.cfg', connect=False, tz_aware=True):
        """
        Constructor for the Mongo class.

        :param conf_file: The configuration file to use for authentication.
        :param connect: The connection flag.
        :param tz_aware: The TimeZone awareness flag.
        """

        # Parse configuration file
        mongo_cfg = Utils.parse_connections_file('mongodb', conf_file)

        host = os.getenv('MONGODB_HOST', mongo_cfg.get('HOST', ''))
        port = os.getenv('MONGODB_PORT', mongo_cfg.get('PORT', ''))
        # add a colon to the port if the port exists
        if port != '':
            port = ':{0}'.format(port)
        user = parse.quote(os.getenv('MONGODB_USER', mongo_cfg.get('USER', '')))
        password = parse.quote(os.getenv('MONGODB_PASSWORD', mongo_cfg.get('PASSWORD', '')))
        database = os.getenv('MONGODB_DATABASE', mongo_cfg.get('DATABASE', ''))
        query_params = os.getenv('MONGODB_QPARAMS', mongo_cfg.get('QUERY_PARAMS', ''))
        # add a question mark to the query parameters if they exist.
        if query_params != '':
            query_params = '?{0}&authSource=admin'.format(query_params)

        self.conn = {}
        self.db = {}

        try:
            self.conn = pymongo.MongoClient("mongodb+srv://{0}:{1}@{2}/{3}?retryWrites=true&w=majority&authSource=admin"
                                            .format(user, password, host, database))
            self.db = self.conn[database]
        except pe.InvalidURI:
            try:
                self.conn = pymongo.MongoClient(
                    'mongodb://{0}:{1}@{2}{3}/{4}{5}'.format(user, password, host, port, database, query_params),
                    connect=connect, tz_aware=tz_aware)
                self.db = self.conn[database]
            except pe.InvalidURI:
                self.conn = pymongo.MongoClient(
                    'mongodb://{0}{1}/{2}{3}'.format(host, port, database, query_params),
                    connect=connect, tz_aware=tz_aware)
                self.db = self.conn[database]

        except pe.ConnectionFailure as e:
            print("Could not connect to MongoDB: {0}".format(e))

    def find(self, collection, query):
        """
        Find elements in a collection.
        
        :param collection: The collection to retrieve the data from.
        :param query: The query to pass to the database.

        :return: The retrieved elements from the database.
        """

        results = list(self.db[collection].find(query))
        return results

    def find_one(self, collection, query):
        """
        Find an element in a collection.

        :param collection: The collection to retrieve the data from.
        :param query: The query to pass to the database.

        :return: The retrieved element from the database.
        """

        result = self.db[collection].find_one(query)
        return result

    def insert_one(self, collection, data):
        """
        Add an element to a collection.

        :param collection: The collection to retrieve the data from.
        :param data: The data to add to the database.

        :return: The added element to the database.
        """

        result = self.db[collection].insert_one(data)
        return result

    def insert_many(self, collection, data):
        """
        Add elements to a collection.

        :param collection: The collection to retrieve the data from.
        :param data: The data to add to the database.

        :return: The added elements to the database.
        """

        result = []
        if len(data) > 0:
            result = self.db[collection].insert_many(data)
        return result

    def replace_one(self, collection, data, query):
        """
        Update or insert data.

        :param collection: The collection to retrieve the data from.
        :param data: The data to update to the database.
        :param query: The filtering query.

        :return: The updated and added elements to the database.
        """
        result = []
        if len(data) > 0:
            result = self.db[collection].replace_one(replacement=data, filter=query, upsert=True)

        return result

    def delete_one(self, collection, query):
        """
        Delete an element to a collection.

        :param collection: The collection to delete the data from.
        :param query: The query to pass to the database.

        :return: The deleted element to the database.
        """

        result = self.db[collection].delete_one(query)
        return result
