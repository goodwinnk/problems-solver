import asyncio
from slack_sdk.web.async_client import AsyncWebClient
from utils.abstract_db_writer import AbstractDbWriter

class DataCollector:
    def __init__(self, db_writer: AbstractDbWriter):
        self.db_writer = db_writer
        self.following_channel_ids = []  # Cached
        self.channels_last_updates = dict()

    async def collect_messages(self, client: AsyncWebClient):
        if not len(self.following_channel_ids):
            self.following_channel_ids = await self.get_following_channels_id()
        for channel_id in self.following_channel_ids:
            last_update = self.db_writer.get_latest_timestamp(channel_id) + 1e-6  # TODO make it more smart
            cursor = None
            while True:
                payload = {"channel": channel_id, "oldest": str(last_update), "limit": 1}  # TODO: limit = ~100-1000~
                if cursor:
                    payload['cursor'] = cursor
                data = await client.conversations_history(**payload)
                if data.get('ok', False):
                    raw_data = data.data
                    print(raw_data)
                    if raw_data['messages']:
                        self.db_writer.add_messages(raw_data['messages'], channel_id)
                    cursor = raw_data['response_metadata']['next_cursor'] if raw_data['has_more'] else None
                elif data.status_code == 429:
                    pause = data.headers['Retry-After']  # TODO: Need testing
                    print(f'OOoops, rate limit. Retry after {pause} s.')
                    await asyncio.sleep(pause)
                if not cursor:
                    break  # DO-WHILE

    async def set_channels(self, checkbox_action: dict):
        channels = dict(
            (checkbox['value'], checkbox['text']['text']) for checkbox in checkbox_action['selected_options'])
        self.following_channel_ids = list(channels.keys())
        current_states = self.db_writer.get_channels_states()
        for channel_id, channel_name in channels.items():
            if channel_id not in current_states:
                current_states[channel_id] = {'name': channel_name, 'id': channel_id,
                                              'following': True, 'last_update': 0}
        for state in current_states.values():
            state['following'] = (state['id'] in channels)
        self.db_writer.update_channels_states(current_states)

    async def get_following_channels_id(self) -> list:
        current = self.db_writer.get_channels_states().values()
        result = []
        for channel in current:
            if channel['following']:
                result.append(channel['id'])
        return result
