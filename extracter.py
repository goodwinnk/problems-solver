"""
Extract data from db to files
"""
import os
import json
from pymongo import MongoClient


def extract():
    mongo = MongoClient()
    if 'problems_solver' in mongo.list_database_names():
        db = mongo['problems_solver']
        os.makedirs('extracted', exist_ok=True)
        collections = db.list_collection_names()
        for collection in collections:
            cursor = db[collection].find()
            data = []
            for document in cursor:
                document['_id'] = str(document['_id'])
                data.append(document)
            json.dump(data, open(f'extracted/{collection}', 'w'), indent=2, ensure_ascii=False)
            print(f"Collection {collection} was dumped")
        print('Completed')
    else:
        print(f"Error: Database 'problems_solver' not found")


if __name__ == "__main__":
    extract()
