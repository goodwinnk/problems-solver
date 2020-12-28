from pymongo import DESCENDING, database

from utils.abstractcontroller import AbstractController


class Controller(AbstractController):
    def __init__(self, db: database.Database):
        self.db = db
        self.channels_states = {}  # channel_id: channel_state

    def get_channels_states(self) -> dict:
        cursor = self.db['channels_states'].find()
        self.channels_states = dict((channel['id'], channel) for channel in cursor)
        return self.channels_states

    def update_channels_states(self, channels_states: dict):
        for channel_id, state in channels_states.items():
            self.channels_states[channel_id] = state  # Make it in two loops in async variant.
            self.db['channels_states'].replace_one({'id': channel_id}, state, upsert=True)

    def add_parent_messages(self, messages: list, channel_id: str):
        if channel_id not in self.db.list_collection_names():
            self.db[channel_id].create_index('ts')
        last_update = self.channels_states[channel_id]['last_update']
        for msg in messages:
            msg['channel_id'] = channel_id
            last_update = max(last_update, float(msg['ts']))
        self.db[channel_id].insert_many(messages)
        self.channels_states[channel_id]['last_update'] = last_update

    def get_parent_messages(self, channel_id: str) -> list:
        return list(self.db[channel_id].find({}))

    def add_dataset_messages(self, channel_id: str, msg_indetifier_pairs: list):
        dataset = self.db["datasets"].find_one({"_id": channel_id})
        dataset = dict((f"{pair[0]}-{pair[1]}", pair) for pair in (dataset['pairs'] if dataset is not None else []))
        dataset.update(dict((f"{pair[0]}-{pair[1]}", pair) for pair in msg_indetifier_pairs))
        dataset = list(dataset.values())
        self.db["datasets"].replace_one({"_id": channel_id}, {"_id": channel_id, "pairs": dataset}, upsert=True)

    def remove_data(self, channel_id: str):
        self.db[channel_id].delete_many({})
        self.db["datasets"].delete_one({"_id": channel_id})

    def get_dataset_data(self, channel_id: str) -> list:
        dataset_doc = self.db["datasets"].find_one({"_id": channel_id})
        return dataset_doc["pairs"] if dataset_doc is not None else []

    def add_private_message(self, message, answer):
        self.db['private_messages'].insert_one({
            "message": message,
            "answer": answer
        })

    def add_child_messages(self, messages: list, channel_id: str, parent_ts: str):
        parent_i = -1
        for i in range(len(messages)):
            if messages[i]['ts'] == parent_ts:
                parent_i = i  # client.conversation_replies also returns parent
        if parent_i != -1:
            messages.pop(parent_i)
        for message in messages:
            message['channel_id'] = channel_id
            self.db[channel_id].find_one_and_update(
                {"ts": float(parent_ts)},
                {"$addToSet": {"thread": message}}
            )

    def get_latest_timestamp(self, channel_id: str) -> float:
        value = self.channels_states[channel_id]['last_update'] if channel_id in self.channels_states else 0
        if not value:
            cursor = self.db[channel_id].find().sort([('ts', DESCENDING)]).limit(1)
            try:
                value = float(cursor.next()['ts'])
            except StopIteration:
                pass
        return value

    def get_private_message(self, direct_id: str, message_ts: str):
        return self.db['private_messages'].find_one({"message.channel": direct_id, "message.ts": message_ts})

    def edit_private_message(self, direct_id: str, message_ts: str, new_msg):
        new_msg['ts'] = message_ts
        new_msg['channel'] = direct_id
        return self.db['private_messages'].find_one_and_update({"message.channel": direct_id, "message.ts": message_ts},
                                                               {"$set": {"message": new_msg}})
        # return self.db['private_messages'].find_one({"message.channel": direct_id, "message.ts": message_ts})
