import os
import logging
from dotenv import load_dotenv

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from data_collector import DataCollector
from mongo_db_writer import MongoDbWriter
from app_home import AppHome

from utils.message_filters import *
from pprint import pprint

load_dotenv('secret.env')
logging.basicConfig(level=logging.INFO)

data_collector = DataCollector(MongoDbWriter())
app_home = AppHome(data_collector)
app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)


@app.event("app_home_opened")
async def draw_home(client: AsyncWebClient, event, logger):
    await client.views_publish(user_id=event['user'], view=await app_home.get_view(client, logger))


@app.action("collect-data")
async def button_clicked(ack, body, client, logger):
    await ack()
    if await data_collector.get_following_channels_ids():
        data_collector.collecting_running = True  # Do it before collecting because we need to draw new interface
        await client.views_update(view_id=body["view"]["id"], hash=body["view"]["hash"],
                                  view=await app_home.get_view(client, logger))
        await data_collector.collect_messages(client, logger)  # A lot of time is wasted here
    await client.views_publish(user_id=body['user']['id'], view=await app_home.get_view(client, logger))


@app.action("following-channel_chosen")
async def choose_channel(ack, body, client, payload, logger):
    await ack()
    await data_collector.set_channels(payload)
    await client.views_update(view_id=body["view"]["id"], hash=body["view"]["hash"],
                              view=await app_home.get_view(client, logger))


@app.message("")
async def message_handler(client: AsyncWebClient, event, message, logger):
    await data_collector.add_message(message)
    if answer_trigger(event):
        await client.chat_postMessage(channel=event['channel'],
                                      thread_ts=get_thread_ts(event),
                                      text='Answer in thread')
    logger.info(message)


@app.event({"type": "message", "subtype": "file_share"})
async def msg_deleted_handler(message, logger):
    logger.info(message)


@app.event({"type": "message", "subtype": "message_deleted"})
async def msg_deleted_handler(message, logger):
    logger.info(message)


@app.event({"type": "message", "subtype": "message_changed"})
async def msg_changed_handler(message, logger):
    logger.info(message)


if __name__ == "__main__":
    app.start(port=3000)
