from google.cloud import translate_v2 as translate
from dotenv import load_dotenv

# !!! ENVIRONMENT VARIABLE REQUIRED
# GOOGLE_APPLICATION_CREDENTIALS=
from nlp.message_processing import text_contain_language

load_dotenv('../secret.env')
translate_client = translate.Client()
# TODO: remove limits
all_counter = 484928


def translate(text: str) -> str:
    global all_counter
    all_counter += len(text)
    print(all_counter)
    if all_counter >= 490000:
        print('limited')
        exit(1)
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
