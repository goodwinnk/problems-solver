import asyncio
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from utils.abstractcontroller import AbstractController
from utils.message_filters import message_contain_russian, translate_message


async def collect_messages(channel_id: str, start_time: float, method, logger, **kwargs):
    """
        method: method to call:
            client.conversation_history
            client.conversations.replies
        kwargs: method args
    """
    cursor = None
    while True:
        payload = {"channel": channel_id, "oldest": str(start_time), "limit": 1000, **kwargs}
        if cursor:  # using pagination
            payload['cursor'] = cursor
        try:
            data = await method(**payload)
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


def translate_messages(messages: list):
    for i in range(len(messages)):
        if message_contain_russian(messages[i]):
            messages[i] = translate_message(messages[i])
    return messages


class DataCollector:
    def __init__(self, controller: AbstractController):
        self.controller = controller
        self.following_channel_ids = []  # Cached
        self.collecting_running = False
        self.data_collected = False

    async def collect_messages(self, client: AsyncWebClient, logger):
        self.collecting_running = True
        self.following_channel_ids = await self.get_following_channels_ids()
        for channel_id in self.following_channel_ids:
            last_update = self.controller.get_latest_timestamp(channel_id) + 1e-6  # web_api includes oldest value
            msg_generator = collect_messages(channel_id, last_update, client.conversations_history, logger)
            async for messages in msg_generator:
                translate_messages(messages)
                self.controller.add_parent_messages(messages, channel_id)
                for message in messages:
                    if 'thread_ts' in message:
                        chd_message_generator = collect_messages(channel_id, last_update,
                                                                 client.conversations_replies, logger,
                                                                 ts=message['thread_ts'])
                        async for child_messages in chd_message_generator:
                            self.controller.add_child_messages(child_messages, channel_id, message['thread_ts'])
            logger.info(f'Channel ID: {channel_id} was scanned')
        self.data_collected = True
        self.collecting_running = False

    async def set_channels(self, checkbox_action: dict):
        channels = dict(
            (checkbox['value'], checkbox['text']['text']) for checkbox in checkbox_action['selected_options'])
        old_following = await self.get_following_channels_ids()
        for channel in channels.keys():
            if channel not in old_following:
                self.data_collected = False
        self.following_channel_ids = list(channels.keys())
        current_states = self.controller.get_channels_states()
        for channel_id, channel_name in channels.items():
            if channel_id not in current_states:
                current_states[channel_id] = {'name': channel_name, 'id': channel_id,
                                              'following': True, 'last_update': 0}
        for state in current_states.values():
            state['following'] = state['id'] in channels
        self.controller.update_channels_states(current_states)

    async def message_from_following_channel(self, message: dict) -> bool:
        return message['channel'] in await self.get_following_channels_ids()

    async def get_following_channels_ids(self) -> list:
        result = []
        current = self.controller.get_channels_states().values()
        for channel in current:
            if channel['following'] and channel['id'] not in result:
                result.append(channel['id'])
        return result

    async def add_message(self, message):
        if await self.message_from_following_channel(message):
            # TODO: maybe parent messages contain 'thread_ts' too.
            if 'thread_ts' in message:
                self.controller.add_child_messages([message], message['channel'], message['thread_ts'])
            else:
                self.controller.add_parent_messages([message], message['channel'])

    async def add_private_message(self, message, answer=None):
        if answer is None:
            answer = "No answer provided"
        self.controller.add_private_message(message, answer)
