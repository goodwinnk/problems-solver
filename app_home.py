from data_collector import DataCollector
from slack_sdk.web.async_client import AsyncWebClient

from utils.block_generator import get_channel_choosing_block, button_block, text_block, header_block


class AppHome:
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector

    async def get_admin_view(self, client: AsyncWebClient, logger):
        is_running = self.data_collector.collecting_running
        subscribed_channels = await self.get_channels_list(client, logger)
        following_channels = await self.data_collector.get_following_channels_ids()
        blocks = []
        if subscribed_channels:
            blocks = get_channel_choosing_block(subscribed_channels, following_channels)
            if not following_channels:
                blocks.append(text_block("Please, choose at least one channel"))
            if not is_running or self.data_collector.data_collected:
                if self.data_collector.data_collected:
                    blocks.append(text_block("The data was collected"))
            else:
                blocks.append(text_block("The scanning process is runned"))
            blocks.append(button_block("Run message sniffer", "click", "collect-data"))
        else:
            blocks.append(text_block("The bot is not subscribed to any channel"))
        return {"type": "home", "blocks": blocks}

    async def get_channels_list(self, client: AsyncWebClient, logger):
        result = []
        try:
            raw_channels = await client.conversations_list()
            if len(raw_channels.get('channels')):
                for channel in raw_channels['channels']:
                    if channel['is_member'] and not channel['is_private']:
                        result.append((channel['name'], channel['id']))
        except Exception as e:
            logger.error(e)
        return result

    def get_view(self):
        blocks = [header_block("Problems solver"),
                  text_block("Ask a question in private messages and we will try to find a similar")]
        return {"type": "home", "blocks": blocks}
