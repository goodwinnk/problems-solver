import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from data_collector import DataCollector
from controller import Controller
from app_home import AppHome
from nlp.message_processing import Message

from nlp.question_detector import is_question
from model_manager import ModelManager
from utils.message_filters import *

load_dotenv('secret.env')
logging.basicConfig(level=logging.INFO)

controller = Controller(MongoClient().get_database('problems_solver'))

model_manager = ModelManager(os.environ.get('MODEL_FOLDER'), controller)
# model_manager.load_from_sources('nlp/data/processed/all_topics.json', 'nlp/data/dataset/production.json', 'UNKNOWN')
model_manager.load_models()

data_collector = DataCollector(controller)
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
    if message_contain_russian(message):
        # logger.info('Contain russian symbols. Translation required')
        # message = translate_message(message)
        # logger.info('Was translated (Actually no)')
        return

    logger.info(message)
    answer = 'Question/problem is not recognized.'
    if is_question(message):
        if await data_collector.message_from_following_channel(message) or message.get('channel_type', 'unk') == 'im':
            if message.get('channel_type', 'unk') == 'im':
                model = model_manager.get_model('UNKNOWN')
            else:
                model = model_manager.get_model(event['channel'])
                await data_collector.add_message(message)
            messages = model.get_similar_messages(Message.from_dict(message))
            logger.info(f"Founded {len(messages)} similar messages")
            if messages:
                answer = '_____________________\n'.join(map(lambda rec: f"{rec[0]}\ntext_sim: {rec[1][0]}\n"
                                                                        f"code_sim: {rec[1][1]}\n"
                                                                        f"entity_sim: {rec[1][2]}\n", messages))
            else:
                answer = 'No similar messages.'
    await client.chat_postMessage(channel=event['channel'],
                                  thread_ts=get_thread_ts(event),
                                  text=answer)
    logger.info(message)


@app.event({"type": "message", "subtype": "file_share"})
async def msg_deleted_handler(message, logger):
    logger.info(message)


@app.event({"type": "message", "subtype": "message_deleted"})
async def msg_file_handler(message, logger):
    logger.info(message)


@app.event({"type": "message", "subtype": "message_changed"})
async def msg_changed_handler(message, logger):
    logger.info(message)


if __name__ == "__main__":
    app.start(port=3000)
