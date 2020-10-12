from pymongo import MongoClient, DESCENDING

from utils.abstract_db_writer import AbstractDbWriter


class MongoDbWriter(AbstractDbWriter):
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.get_database('problems_solver')
        self.channels_states = {}  # channel_id: channel_state

    def get_channels_states(self) -> dict:
        if not self.channels_states:
            cursor = self.db['channels states'].find()
            self.channels_states = dict((channel['id'], channel) for channel in cursor)
        return self.channels_states

    def update_channels_states(self, channels_states: dict):
        for channel_id, state in channels_states.items():
            self.channels_states[channel_id] = state  # Make it in two loops in async variant.
            self.db['channels states'].replace_one({'id': channel_id}, state, upsert=True)

    def add_messages(self, messages: list, channel_id: str):
        if channel_id not in self.db.list_collection_names():
            self.db[channel_id].create_index('ts')
        last_update = self.channels_states[channel_id]['last_update']
        for msg in messages:
            msg['ts'] = float(msg['ts'])
            last_update = max(last_update, msg['ts'])
        self.db[channel_id].insert_many(messages, ordered=True)  # Are you sure? (ordered)
        self.channels_states[channel_id]['last_update'] = last_update

    def get_latest_timestamp(self, channel_id: str) -> float:
        value = self.channels_states[channel_id]['last_update'] if channel_id in self.channels_states else 0
        if not value:
            cursor = self.db[channel_id].find().sort([('ts', DESCENDING)]).limit(1)
            try:
                value = cursor.next()['ts']
            except StopIteration:
                print(f'Chanel {channel_id} is empty')
        return value
