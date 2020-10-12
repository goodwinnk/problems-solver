def get_channel_choosing_block(raw_channels: list, following_ids: list) -> list:
    """
    filtering channel list and product Block kit Form
    :param channels: list of pairs: ( channel_id, channel_name )
    :return: list: Block kit interpretation of channel choosing
    """
    blocks = [{
        "type": "section", "text": {
            "type": "mrkdwn", "text": "The bot is subscribed to these channels, choose which ones to follow:"
        },
        "accessory": {"type": "checkboxes", "options": [], "action_id": "following-channel_chosen"}},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Run message sniffer", "emoji": False},
             "value": "click_me_123", "style": "primary", "action_id": "collect-data"}]
         }
    ]
    for channel in raw_channels:
        if channel['is_member'] and not channel['is_private']:
            option = {
                "text": {
                    "type": "mrkdwn",
                    "text": channel['name']
                },
                "value": channel['id']
            }
            blocks[0]["accessory"]["options"].append(option)
            if channel['id'] in following_ids:
                if 'initial_options' not in blocks[0]['accessory']:
                    blocks[0]['accessory']['initial_options'] = []
                    # We can't create this list before because all following channels may be removed
                blocks[0]['accessory']['initial_options'].append(option)
    return blocks
