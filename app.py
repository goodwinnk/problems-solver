import os
import logging

from dotenv import load_dotenv
from pymongo import MongoClient

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from data_collector import DataCollector
from controller import Controller
from app_home import AppHome
from message_handler import send_answers
from nlp.message_processing import Message

from model_manager import ModelManager
from utils.message_filters import *

load_dotenv('secret.env')
logging.basicConfig(level=logging.INFO)

controller = Controller(MongoClient().get_database('problems_solver'))

model_manager = ModelManager(os.environ.get('MODEL_FOLDER'), controller)
model_manager.load_from_sources('nlp/data/processed/all_topics.json', 'nlp/data/dataset/production.json', 'C4U955N6B')
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
        logging.info('Start collecting messages')
        data_collector.collecting_running = True  # Do it before collecting because we need to draw new interface
        await client.views_update(view_id=body["view"]["id"], hash=body["view"]["hash"],
                                  view=await app_home.get_view(client, logger))
        await data_collector.collect_messages(client, logger)  # A lot of time is wasted here
    model_manager.load_models(from_files=False)
    await client.views_publish(user_id=body['user']['id'], view=await app_home.get_view(client, logger))


def update_dataset(first_key: str, second_key: str, status: bool):
    logging.info(f'Updating dataset:{first_key}-{second_key}: {status}')
    f_channel_id, s_channel_id = first_key.split('-')[0], second_key.split('-')[0]
    if f_channel_id == s_channel_id:
        controller.add_dataset_messages(channel_id=f_channel_id, msg_indetifier_pairs=[[first_key, second_key, status]])
        logging.info('Dataset was updated')
    else:
        logging.warning('Review message pair from different channels!')


@app.action("review-positive")
async def review_positive(ack, action):
    await ack()
    logging.info('We have new POSITIVE review')
    first, second = action['value'].split('/')
    update_dataset(first, second, True)


@app.action("review-negative")
async def review_negative(ack, action):
    await ack()
    logging.info('We have new NEGATIVE review')
    first, second = action['value'].split('/')
    update_dataset(first, second, False)


@app.action("following-channel_chosen")
async def choose_channel(ack, body, client, payload, logger):
    await ack()
    await data_collector.set_channels(payload)
    # await client.views_update(view_id=body["view"]["id"], hash=body["view"]["hash"],
    #                           view=await app_home.get_view(client, logger))


async def answer_handler(client: AsyncWebClient, event, message):
    channel = message['channel']
    logging.info(f"Following channels: {await data_collector.get_following_channels_ids()}")
    if message.get('channel_type') == 'im':
        logging.info("'Im' message received")
        model = model_manager.get_model('C01CBLSMX0V')  # default channel to answer
        answers = await send_answers(client, model, event, message)
        await data_collector.add_private_message(message, answers)
    elif channel in await data_collector.get_following_channels_ids():
        logging.info('Message from following channel received')
        model = model_manager.get_model(channel)
        await send_answers(client, model, event, message)
        await data_collector.add_message(message)
        model.update_model(Message.from_dict(message))
        model_manager.save_models()


@app.message("")
async def message_handler(client: AsyncWebClient, event, message, logger):
    logger.info(message)
    if message_contain_russian(message):
        logger.info('Message was ignored: contain russian')
        return
    await answer_handler(client, event, message)


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
