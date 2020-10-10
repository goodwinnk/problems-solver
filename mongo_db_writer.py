from datetime import datetime
from pymongo import MongoClient, DESCENDING

from utils.abstract_db_writer import AbstractDbWriter


class MongoDbWriter(AbstractDbWriter):
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.get_database('problems_solver')
        self.last_updates = {}  # channel_id: timestamp

    def get_channels_states(self) -> dict:
        channels = {}
        cursor = self.db['channels states'].find()
        for channel in cursor:
            if channel['id'] not in self.last_updates:
                self.last_updates[channel['id']] = channel['last_update'] = self.get_latest_timestamp(channel['id'])
            channels[channel['id']] = channel
        return channels

    def update_channels_states(self, channels_states: dict):
        for state in channels_states.values():
            self.db['channels states'].replace_one({'id': state['id']}, state, upsert=True)

    def add_messages(self, messages: list, channel_id: str):
        if channel_id not in self.db.list_collection_names():
            self.db[channel_id].create_index('ts')
        last_update = 0
        for msg in messages:
            msg['ts'] = float(msg['ts'])
            last_update = max(last_update, msg['ts'])
        self.db[channel_id].insert_many(messages, ordered=True)  # Are you sure? (ordered)
        self.last_updates[channel_id] = max(self.last_updates.get(channel_id, 0), last_update)

    def get_latest_timestamp(self, channel_id: str) -> float:
        value = self.last_updates.get(channel_id, 0)
        if not value:
            cursor = self.db[channel_id].find().sort([('ts', DESCENDING)]).limit(1)
            try:
                value = cursor.next()['ts']
            except StopIteration:
                print(f'Value not founded {channel_id}')
        return value
