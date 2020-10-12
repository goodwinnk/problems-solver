def get_channel_choosing_block(subscribed_channels: list, followed_channels: list) -> list:
    """
    filtering channel list and product Block kit Form
    :param channels: list of pairs: ( channel_id, channel_name )
    :return: list: Block kit interpretation of channel choosing
    """
    blocks = [{
        **get_text("The bot is subscribed to these channels, choose which ones to follow:"),
        "accessory": {"type": "checkboxes", "options": [], "action_id": "following-channel_chosen"}}
    ]
    for channel_name, channel_id in subscribed_channels:
        option = {
            "text": {
                "type": "mrkdwn",
                "text": channel_name
            },
            "value": channel_id
        }
        blocks[0]["accessory"]["options"].append(option)
        if channel_id in followed_channels:
            if 'initial_options' not in blocks[0]['accessory']:
                blocks[0]['accessory']['initial_options'] = []
                # We can't create this list before because all following channels may be removed
            blocks[0]['accessory']['initial_options'].append(option)
    return blocks


def get_text(text: str):
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": text
        }
    }


def get_button(text: str, value: str, action_id: str):
    return {
        "type": "actions",
        "elements": [{
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": False
            },
            "value": value,
            "style": "primary",
            "action_id": action_id
        }]
    }
