import logging
from copy import deepcopy

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from nlp.message_processing import Message
from nlp.model import Model
from nlp.question_detector import is_question
from utils.message_filters import get_thread_ts

template = {
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "We found a similar <{}|message>, please rate!"
            }
        },
        {
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Revelant"},
                 "style": "primary", "action_id": "review-positive", "value": ""
                 },
                {
                    "type": "button", "text": {"type": "plain_text", "text": "Useless"},
                    "style": "danger", "action_id": "review-negative", "value": ""
                }
            ]
        }
    ]
}


async def send_answers(client: AsyncWebClient, model: Model, event: dict, message: dict) -> list:
    answers = []
    channel, thread_ts = message['channel'], get_thread_ts(event)
    if is_question(message):
        message = Message.from_dict(message)
        messages = model.get_similar_messages(message)
        logging.info(f"Founded {len(messages)} similar messages")
        if messages:
            for sim_msg, sim in messages:
                logging.info(f'Create permalink for: {sim_msg.channel_id}-{sim_msg.ts}')
                try:
                    link_response = await client.chat_getPermalink(channel=sim_msg.channel_id, message_ts=sim_msg.ts)
                except SlackApiError:
                    link_response = {'ok': False}
                if link_response['ok']:
                    new_answer = deepcopy(template)
                    temp = new_answer['blocks'][0]['text']['text']
                    new_answer['blocks'][0]['text']['text'] = temp.format(link_response['permalink'])
                    new_answer['blocks'][1]['elements'][0]['value'] = f"{message.get_key()}/{sim_msg.get_key()}"
                    new_answer['blocks'][1]['elements'][1]['value'] = f"{message.get_key()}/{sim_msg.get_key()}"
                    await client.chat_postMessage(channel=channel, thread_ts=thread_ts, unfurl_links=True,
                                                  text=f"<{link_response['permalink']}>", blocks=new_answer['blocks'])
                    answers.append(new_answer)
        else:
            answers.append({'blocks': [], 'text': 'No similar messages.'})
            await client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                          text='No similar messages.')
    else:
        answers.append({'blocks': [], 'text': "Question is not detected. Try add '?'"})
        await client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                      text="Question is not detected. Try add '?'")
    return answers
