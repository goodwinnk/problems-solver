import os
from pprint import pprint

from dotenv import load_dotenv

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from data_collector import DataCollector
from mongo_db_writer import MongoDbWriter
from utils.block_generator import get_channel_choosing_block

load_dotenv('secret.env')

data_collector = DataCollector(MongoDbWriter())
app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)


@app.action("collect-data")
async def button_clicked(ack, say, client, logger):
    await ack()
    if await data_collector.get_following_channels_ids():
        if not data_collector.collecting_running:
            await say('Start collection messages')
            await data_collector.collect_messages(client, logger)
            await say('New messages from following channels was stored.')
        else:
            await say('The scanning process is already runned')
    else:
        await say('Please choose the channels')


@app.action("following-channel_chosen")
async def choose_channel(ack, payload):
    await ack()
    await data_collector.set_channels(payload)


@app.command('/collect')
async def collect_messages(ack, say, client: AsyncWebClient, logger):
    await ack()
    result = await client.conversations_list()
    if result.get('ok'):
        if not len(result.get('channels')):
            await say('No channels found')
        else:
            following = await data_collector.get_following_channels_ids()
            blocks = get_channel_choosing_block(result.get('channels'), following)
            await say(blocks=blocks)
    else:
        await say(f'Something goes wrong: {str(result)}')
        logger.error()


@app.message("")
async def message_handler(message, say, client):
    await say(f"Hey there <@{message['user']}>!")
    await data_collector.add_message(message)


@app.event({"type": "message", "subtype": "message_deleted"})
async def msg_deleted_handler(message, say):
    print(message)
    await say('Oh, message was deleted.')


@app.event({"type": "message", "subtype": "message_changed"})
async def msg_changed_handler(message, say):
    print(message)
    await say('The message was changed.')


if __name__ == "__main__":
    app.start(port=3000)
