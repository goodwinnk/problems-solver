import asyncio
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
from utils.abstract_db_writer import AbstractDbWriter


async def collect_messages(channel_id: str, start_time: float, client: AsyncWebClient, logger):
    cursor = None
    while True:
        payload = {"channel": channel_id, "oldest": str(start_time), "limit": 1000}
        if cursor:
            payload['cursor'] = cursor
        try:
            data = await client.conversations_history(**payload)
            raw_data = data.data
            if raw_data['messages']:
                yield raw_data['messages']
            cursor = raw_data['response_metadata']['next_cursor'] if raw_data['has_more'] else None
        except SlackApiError as e:
            if e.response['error'] == 'ratelimited':
                pause = float(e.response.headers['retry-after'])
                logger.info(f'Rate limit exceeded, sleep for {pause} s.')
                await asyncio.sleep(pause)
            else:
                logger.error(f'SlackApiError {e.response.status_code}:\n {e.response}')
        if not cursor:
            break  # DO-WHILE


class DataCollector:
    def __init__(self, db_writer: AbstractDbWriter):
        self.db_writer = db_writer
        self.following_channel_ids = []  # Cached
        self.collecting_running = False
        self.data_collected = False

    async def collect_messages(self, client: AsyncWebClient, logger):
        self.collecting_running = True
        self.following_channel_ids = await self.get_following_channels_ids()
        for channel_id in self.following_channel_ids:
            last_update = self.db_writer.get_latest_timestamp(channel_id) + 1e-6  # web_api includes oldest value
            msg_generator = collect_messages(channel_id, last_update, client, logger)
            async for messages in msg_generator:
                self.db_writer.add_messages(messages, channel_id)
            logger.info(f'Channel ID: {channel_id} was scanned')
        self.data_collected = True
        self.collecting_running = False

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
            state['following'] = state['id'] in channels
        self.db_writer.update_channels_states(current_states)

    async def get_following_channels_ids(self) -> list:
        result = self.following_channel_ids
        if not result:
            current = self.db_writer.get_channels_states().values()
            for channel in current:
                if channel['following']:
                    result.append(channel['id'])
        return result

    async def add_message(self, message):
        if message['channel'] in await self.get_following_channels_ids():
            self.db_writer.add_messages([message], message['channel'])
