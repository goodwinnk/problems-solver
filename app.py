import os
import logging

from dotenv import load_dotenv
from pymongo import MongoClient

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from data_collector import DataCollector
from controller import Controller
from app_home import AppHome
from message_handler import send_answers, forward_btn_show
from nlp.message_processing import Message

from model_manager import ModelManager
from utils.message_filters import *

load_dotenv('secret.env')
logging.basicConfig(level=logging.INFO)
admin_id = os.environ.get('ADMIN_ID')
default_channel_id = os.environ.get('CHANNEL_ID')
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
    if event['user'] == admin_id:
        await client.views_publish(user_id=event['user'], view=await app_home.get_admin_view(client, logger))
    else:
        await client.views_publish(user_id=event['user'], view=app_home.get_view())


@app.action("collect-data")
async def button_clicked(ack, body, client, logger):
    await ack()
    if await data_collector.get_following_channels_ids():
        logging.info('Start collecting messages')
        data_collector.collecting_running = True  # Do it before collecting because we need to draw new interface
        await client.views_update(view_id=body["view"]["id"], hash=body["view"]["hash"],
                                  view=await app_home.get_admin_view(client, logger))
        await data_collector.collect_messages(client, logger)  # A lot of time is wasted here
    logging.info("Reloading models")
    model_manager.load_models(from_files=False)
    await client.views_publish(user_id=body['user']['id'], view=await app_home.get_admin_view(client, logger))


def update_dataset(first_key: str, second_key: str, status: bool):
    logging.info(f'Try to update dataset:{first_key}-{second_key}: {status}')
    f_channel_id, s_channel_id = first_key.split('-')[0], second_key.split('-')[0]
    if f_channel_id == s_channel_id:
        controller.add_dataset_messages(channel_id=f_channel_id, msg_indetifier_pairs=[[first_key, second_key, status]])
        logging.info('Dataset was updated')
    else:
        logging.warning('Review message pair from different channels! Dataset was not updated.')


@app.action("review-positive")
async def review_positive(ack, action):
    logging.info('We have new POSITIVE review')
    first, second = action['value'].split('/')
    update_dataset(first, second, True)
    await ack()


@app.action("review-negative")
async def review_negative(ack, action):
    logging.info('We have new NEGATIVE review')
    first, second = action['value'].split('/')
    update_dataset(first, second, False)
    await ack()


@app.action("forward-message")
async def forward_message(ack, client, action):
    logging.info('Forwarding message to main channel')
    direct_id, msg_ts = action['value'].split('-')
    msg = controller.get_private_message(direct_id, msg_ts)
    if msg is not None:
        text = msg['message']['text']
        user = msg['message']['user']
        text += f'\nAuthor: <@{user}>'
        await client.chat_postMessage(channel=default_channel_id, link_names=True, text=text)
    await ack()


@app.action("following-channel_chosen")
async def choose_channel(ack, payload):
    await data_collector.set_channels(payload)
    await ack()


async def answer_handler(client: AsyncWebClient, event, message):
    channel = message['channel']
    logging.info(f"Following channels: {await data_collector.get_following_channels_ids()}")
    if message.get('channel_type') == 'im':
        logging.info("'Im' message received")
        model = model_manager.get_model(default_channel_id)  # default channel to answer
        answers, thread_ts = await send_answers(client, model, event, message)
        await forward_btn_show(client, channel, message['ts'], thread_ts)
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
    logging.info(message)
    if message_contain_russian(message):
        logging.info('Message was ignored: contain russian')
        return
    await answer_handler(client, event, message)


@app.event({"type": "message", "subtype": "file_share"})
async def msg_deleted_handler():
    pass


@app.event({"type": "message", "subtype": "message_deleted"})
async def msg_file_handler():
    pass


@app.event({"type": "message", "subtype": "message_changed"})
async def msg_changed_handler(message):
    if message.get('channel_type') == 'im':
        logging.info('User change private message')
        ts = message['previous_message']['ts']
        controller.edit_private_message(message['channel'], ts, message['message'])


if __name__ == "__main__":
    app.start(port=3000)
