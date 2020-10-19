import copy

russian_symbols = 'абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'


# from google.cloud import translate_v2 as translate
# translate_client = translate.Client()


def answer_trigger(message) -> bool:
    if 'thread_answer' in message['text'].lower():
        return True
    return False


def get_thread_ts(event):
    if 'thread_ts' in event:
        return event['thread_ts']
    return event['event_ts']  # because event['ts'] it's not string! Maybe it's an error.


def text_contain_russian(text: str) -> bool:
    rus = set(s for s in russian_symbols)
    for symbol in text:
        if symbol in rus:
            return True
    return False


def message_contain_russian(message):
    return text_contain_russian(message['text'])


def translate_message(message: dict) -> dict:
    # TODO: Decide whether to translate the 'text' field in all cases
    # TODO: we are translating twice, first time - text, second - time text, so make it smarter
    message['original_text'] = copy.copy(message['text'])
    message['text'] = translate_text(message['text'])
    if 'blocks' in message:
        message['original_blocks'] = copy.deepcopy(message['blocks'])
        for block in message['blocks']:
            if block['type'] == 'rich_text':
                for element in block['elements']:
                    if element['type'] == 'rich_text_section':
                        for inner_element in element['elements']:
                            if isinstance(inner_element.get('text', None), str) and \
                                    text_contain_russian(inner_element['text']):
                                inner_element['text'] = translate_text(inner_element['text'])
    return message


def translate_text(text: str) -> str:
    # # !This method require specified GOOGLE_APPLICATION_CREDENTIALS environment variable!
    # When it will be available uncomment it and lines on top of file
    # result = translate_client.translate(text, source_language='ru', target_language='en')
    # return result["translatedText"]
    return f'ENGLISH<{text}>'
