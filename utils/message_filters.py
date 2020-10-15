def answer_trigger(event) -> bool:
    print(event)
    if 'thread_answer' in event['text'].lower():
        return True
    return False


def get_thread_ts(event):
    if 'thread_ts' in event:
        return event['thread_ts']
    return event['event_ts']  # because event['ts'] it's not string! Maybe it's an error.
