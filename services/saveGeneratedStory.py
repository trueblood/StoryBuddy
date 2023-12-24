from pymongo import MongoClient

class MongoDBClient:
    def __init__(self, host, port, database, collection):
        self.client = MongoClient(host, port)
        self.db = self.client[database]
        self.collection = self.db[collection]

    def save_data(self, data):
        self.collection.insert_one(data)
        print("Data saved successfully!")

    def close_connection(self):
        self.client.close()
        print("Connection closed.")