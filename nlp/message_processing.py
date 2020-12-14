import os
import json
import glob
from collections import Counter

rus_letters = 'абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'
eng_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
all_letters = rus_letters + eng_letters + '0123456789'


def text_contain_language(text: str, russian=True) -> bool:
    letters = rus_letters if russian else eng_letters
    lang = set(s for s in letters)
    for symbol in text:
        if symbol in lang:
            return True
    return False


def message_contain_russian(message):
    return text_contain_language(message['text'])


class Message:
    def __init__(self, text: str, channel_id: str, ts: str, author_id: str):
        self.text = text
        self.author_id = author_id
        self.channel_id = channel_id
        self.ts = ts

    @classmethod
    def from_dict(cls, dict_data: dict):
        channel_id = dict_data.get('channel', dict_data.get('channel_id')) # dict_data.get('channel_id', 'UNKNOWN')
        author_id = dict_data.get('user', dict_data.get('bot_id', 'UNKNOWN'))
        return cls(dict_data['text'], channel_id, dict_data['ts'], author_id)

    def get_key(self):
        return f'{self.channel_id}-{self.ts}'

    def __cmp__(self, other):
        return self.channel_id == other.channel_id and self.ts == other.ts

    def __str__(self):
        return self.text


def read_data(file_or_folder_path: str):
    if os.path.isdir(file_or_folder_path):
        result = []
        files_list = glob.glob(f"{file_or_folder_path}/*.json")
        print(f"Founded {len(files_list)} files")
        for file in files_list:
            result.extend(json.load(open(file, 'r', encoding='utf-8')))
        print(f"Total {len(result)} messages")
        return result
    else:
        return json.load(open(file_or_folder_path, 'r', encoding='utf-8'))


def group_threads(messages: list):
    result = dict()
    unhandled_, system = [], []

    def add_message(result, message_):
        for reply in result[message_['thread_ts']]['replies']:
            if reply['ts'] == message_['ts']:
                reply['message'] = message_
                return

    for message in messages:
        if 'replies' in message:
            result[message['thread_ts']] = message
        elif 'thread_ts' in message:
            if message['thread_ts'] in result:
                add_message(result, message)
            else:
                unhandled_.append(message)
        elif 'subtype' in message:
            system.append(message)
        else:
            result[message['ts']] = message

    for message in unhandled_:
        if 'thread_ts' in message:
            if message['thread_ts'] in result:
                add_message(result, message)
                continue
        result[message['ts']] = message
    return result, system


def parse_text(raw_text):
    parsed, current_part, inner_text = [], [], []
    link_status, code_status, quoted_status, emoji_status = 0, 0, 0, 0
    text = f'###{raw_text}###'
    for i, symbol in zip(range(3, 3 + len(raw_text)), text[3:-3]):
        # Code extracting part
        if symbol == '`' and text[i:i + 3] == '```' and code_status == 0:
            code_status = 1
            if current_part:
                parsed.append({'type': 'text', 'text': ''.join(current_part)})
            current_part.clear()
            current_part.append(symbol)
        elif symbol == '`' and text[i - 2:i + 1] == '```' and code_status != 0:
            if code_status == 1:
                code_status = 2
                current_part.append(symbol)
            else:
                current_part.append(symbol)
                parsed.append({'type': 'code', 'text': ''.join(current_part[3:-3]), 'full': ''.join(current_part)})
                current_part.clear()
                code_status = 0
        elif code_status != 0:
            current_part.append(symbol)

        # Links extracting part
        elif symbol == '<':
            link_status = 1
            if current_part:
                parsed.append({'type': 'text', 'text': ''.join(current_part)})
            current_part.clear()
            current_part.append(symbol)
        elif link_status != 0:
            current_part.append(symbol)
            if symbol == '|':
                inner_text.clear()
                link_status = 2
            elif symbol == '>':
                parsed.append({'type': 'link', 'text': ''.join(inner_text), 'full': ''.join(current_part)})
                current_part.clear()
                inner_text.clear()
                link_status = 0
            elif link_status == 2:
                inner_text.append(symbol)

        # Parse quoted text:
        elif symbol == '`':
            if quoted_status == 0:
                quoted_status = 1
                if current_part:
                    parsed.append({'type': 'text', 'text': ''.join(current_part)})
                current_part.clear()
                current_part.append(symbol)
            else:
                current_part.append(symbol)
                parsed.append({'type': 'quoted', 'text': ''.join(current_part[1:-1]), 'full': ''.join(current_part)})
                quoted_status = 0
                current_part.clear()
        elif quoted_status == 1:
            current_part.append(symbol)

        # Parse emoji
        elif symbol == ':':
            if emoji_status == 0 and text_contain_language(text[i + 1], russian=False):
                if current_part:
                    parsed.append({'type': 'text', 'text': ''.join(current_part)})
                    current_part.clear()
                emoji_status = 1
                current_part.append(symbol)
            elif emoji_status == 1:
                current_part.append(symbol)
                parsed.append({'type': 'emoji', 'text': ''.join(current_part[1:-1]), 'full': ''.join(current_part)})
                current_part.clear()
                emoji_status = 0
            else:
                current_part.append(symbol)
        elif emoji_status == 1:
            if not text_contain_language(symbol, russian=False) and symbol not in '_-':
                emoji_status = 0
            current_part.append(symbol)

        # Just text
        else:
            current_part.append(symbol)

    if current_part:
        parsed.append({'type': 'text', 'text': ''.join(current_part)})
    return parsed


def unparse_text(parsed: list) -> str:
    result = []
    for elem in parsed:
        if elem['type'] == 'text':
            result.append(elem['text'])
        else:
            result.append(elem['full'])
    return ''.join(result)


def extract_tokens(parsed: list, lower=True, targets=('text', 'quoted', 'link')):
    tokens, letters = [], set(all_letters)

    def add_token(token_: list):
        if token_:
            parts = (''.join(token_) if not lower else ''.join(token_).lower()).split('/')
            tokens.extend(parts)
            token_.clear()

    for item in parsed:
        token = []
        if item['type'] in targets:
            text_ = f"#{item['text']}#"
            for i, symbol in zip(range(1, len(text_)), text_[1:-1]):
                if symbol in letters:
                    token.append(symbol)
                elif symbol == '.':
                    if token and (text_[i + 1] in letters or (len(token) > 2 and token[-2] == '.')):
                        token.append(symbol)
                    else:
                        add_token(token)
                elif symbol in ' ,:;*?!()[]{}<>\\"\'+=~#$&|^\r\n\t':
                    add_token(token)
                elif token and symbol in '-_`/' and text_[i + 1] in letters:
                    token.append(symbol)
        add_token(token)
    return tokens


def frequency_analysis(messages, stemmer, stopwords: set):
    result = Counter()
    for message in messages:
        stemmed_tokens = []
        for token in extract_tokens(parse_text(message['text'])):
            if token not in stopwords and stemmer.stem(token) not in stopwords:
                stemmed_tokens.append(stemmer.stem(token))
        result.update(stemmed_tokens)
    return result
