import os
import json
import glob
from google.cloud import translate_v2 as translate
from collections import Counter
from dotenv import load_dotenv
from pprint import pprint

# !!! ENVIRONMENT VARIABLE REQUIRED
# GOOGLE_APPLICATION_CREDENTIALS=
# !!!
load_dotenv('../secret.env')
translate_client = translate.Client()

rus_letters = 'абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'
eng_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
all_letters = rus_letters + eng_letters


def text_contain_language(text: str, russian=True) -> bool:
    letters = rus_letters if russian else eng_letters
    lang = set(s for s in letters)
    for symbol in text:
        if symbol in lang:
            return True
    return False


def message_contain_russian(message):
    return text_contain_language(message['text'])


def read_data(file_or_folder_path: str) -> list:
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
    unhandled_ = []
    unknown, system = [], []

    def add_message(result, message_):
        for reply in result[message_['thread_ts']]['replies']:
            if reply['ts'] == message_['ts']:
                reply['message'] = message_

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
            unknown.append(message)

    for message in unhandled_:
        if 'thread_ts' in message:
            if message['thread_ts'] in result:
                add_message(result, message)
                continue
        unknown.append(message)
    return result, unknown, system


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
            for i, symbol in zip(range(2, len(text_)), text_[2:-1]):
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


def translate(text: str) -> str:
    result = translate_client.translate(text, source_language='ru', target_language='en', format_='text')
    return result["translatedText"]


def translate_parsed_text(parsed: list):
    for elem in parsed:
        if elem['type'] in ['text', 'quoted', 'link']:
            if text_contain_language(elem['text']):
                elem['origin'] = elem['text'] if elem['type'] == 'text' else elem['full']
                elem['text'] = translate(elem['text'])
                if elem['type'] == 'text':
                    elem['full'] = elem['text']
                elif elem['type'] == 'quoted':
                    elem['full'] = f'`{elem["text"]}`'
                elif elem['type'] == 'link':
                    link = elem['origin'][1:elem['origin'].find('|')]
                    elem['full'] = f'<{link}|{elem["text"]}>'
                elif elem['type'] == 'code':
                    elem['full'] = f"```{elem['text']}```"
    return parsed


def frequency_analysis(messages):
    result = Counter()
    for message in messages:
        result.update(extract_tokens(parse_text(message['text'])))
    return result


def translate_topics():
    data = read_data("./data/kotlin_year")
    groups_dict, _, _ = group_threads(data)
    russian, english = [], []
    for message in data:
        if message_contain_russian(message):
            russian.append(message)
        else:
            english.append(message)
    russian_topics = list(filter(lambda x: message_contain_russian(x), groups_dict.values()))
    ext_translations, translations = [], []
    for i, message in zip(range(len(russian_topics)), russian_topics):
        parsed = parse_text(message['text'])
        print(i)
        translated = translate_parsed_text(parsed)
        new_ext_message = {
            'origin': message['text'],
            'ts': message['ts'],
            'user': message['user'],
            'translated': translated,
            'text': unparse_text(translated)
        }
        new_message = {
            "text": new_ext_message['text'],
            "origin": new_ext_message['origin']
        }
        ext_translations.append(new_ext_message)
        translations.append(new_message)
        json.dump(ext_translations, open('data/translations/parsed_topics.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)
        json.dump(translations, open('data/translations/only_text_topics.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)


def read_corpus(path) -> list:
    file = open(path, 'r')
    data = file.read().split('\n')
    result = []
    for item in data:
        item = item.split('\t')
        if len(item) == 2:
            result.append((item[1], int(item[0])))
    del data
    return result


def create_corpus(messages):
    stats = frequency_analysis(messages)
    return sorted(list((key, value) for key, value in stats.items()), key=lambda x: x[1], reverse=True)


def save_corpus(corpus: list, path: str):
    with open(path, 'w') as file:
        file.write('\n'.join([f"{count}\t{token}" for token, count in corpus]))


def get_corpus_diff(first_corpus: list, second_corpus: list, freq_coef, lower=True) -> list:
    first_dict = dict(map(lambda x: (x[0].lower(), x[1]), first_corpus))
    second_dict = dict(map(lambda x: (x[0].lower(), x[1]), second_corpus))
    first_total = sum(map(lambda x: x[1], first_corpus))
    second_total = sum(map(lambda x: x[1], second_corpus))
    slang_tokens = set()

    for f_token, f_token_count in first_dict.items():
        if f_token_count / first_total > freq_coef * second_dict.get(f_token.lower(), 0) / second_total:
            slang_tokens.add(f_token)
    return list(filter(lambda x: x[0] in slang_tokens, first_corpus))


def main():
    all_messages = read_data("data/kotlin_year")
    groups_dict, unknown_msg, system_msg = group_threads(all_messages)
    print(f"Grouped into {len(groups_dict)} threads")
    print(f"Founded {len(unknown_msg)} unthreaded\nFounded {len(system_msg)} system messages")
    russian, english = [], []
    for message in all_messages:
        if message_contain_russian(message):
            russian.append(message)
        else:
            english.append(message)
    print(f"EN: {len(english)}\nRU: {len(russian)}")

    eng_translations = read_data('data/translations/only_text_topics.json')
    true_rus_corpus = read_corpus('data/corpus/true-rus-corpus.txt')
    translation_corpus = create_corpus(eng_translations)
    eng_corpus = create_corpus(english)
    rus_corpus = create_corpus(russian)
    save_corpus(eng_corpus, 'data/corpus/eng-corpus.txt')
    save_corpus(rus_corpus, 'data/corpus/rus-corpus.txt')
    save_corpus(translation_corpus, 'data/corpus/translation-corpus.txt')

    eng_total = sum(map(lambda x: x[1], eng_corpus))
    rus_total = sum(map(lambda x: x[1], rus_corpus))
    translate_total = sum(map(lambda x: x[1], translation_corpus))
    print("Total english words:", eng_total)
    print("Total russian words:", rus_total)
    print("Translated words:", translate_total)

    rus_slang_corpus = get_corpus_diff(rus_corpus, true_rus_corpus, 10000000)
    translation_slang_corpus = get_corpus_diff(translation_corpus, eng_corpus, 5)
    save_corpus(rus_slang_corpus, 'data/corpus/rus-slang-corpus.txt')
    save_corpus(translation_slang_corpus, 'data/corpus/translation-slang-corpus.txt')

    # View all messages containing the token:
    token = 'hello'
    for message in all_messages:
        if token in message['text'].lower():
            pprint(message)
            # Only text: print(message['text'])


if __name__ == "__main__":
    main()
