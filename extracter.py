"""
Extract data from db to files
"""
import os
import json
from pymongo import MongoClient


def reformat_document(document):
    new_document = dict()
    if 'user' in document:
        new_document['user'] = document['user']
        new_document['text'] = document['text']
        if 'original_text' in document:
            new_document['original'] = document.get('original_text')
        if 'thread' in document:
            new_document['thread'] = [reformat_document(answer) for answer in document['thread']]
        if 'reactions' in document:
            new_document['reaction'] = [reaction['name'] for reaction in document['reactions']]

        return new_document
    document['_id'] = str(document['_id'])
    return document


def extract():
    mongo = MongoClient()
    if 'problems_solver' in mongo.list_database_names():
        db = mongo['problems_solver']
        os.makedirs('extracted', exist_ok=True)
        collections = db.list_collection_names()
        for collection in collections:
            cursor = db[collection].find()
            data = [reformat_document(document) for document in cursor]
            json.dump(data, open(f'extracted/{collection}', 'w'), indent=2, ensure_ascii=False)
            print(f"Collection {collection} was dumped")
        print('Completed')
    else:
        print(f"Error: Database 'problems_solver' not found")


if __name__ == "__main__":
    extract()
