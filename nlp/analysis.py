import json

from nltk import PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from nlp.translator import translate_parsed_text
from nlp.corpus import read_corpus, create_corpus, save_corpus, get_corpus_diff
from nlp.message_processing import read_data, parse_text, unparse_text, message_contain_russian, group_threads


def translate_messages(messages: list, output_name):
    global all_counter
    russian_messages = list(filter(lambda x: message_contain_russian(x), messages))[1503:]
    ext_translations, translations = [], []
    for i, message in zip(range(len(russian_messages)), russian_messages):
        parsed = parse_text(message['text'])
        print(i, all_counter + len(message['text']))
        translated = translate_parsed_text(parsed)
        new_ext_message = {
            'origin': message['text'],
            'ts': message['ts'],
            'thread_ts': message.get('thread_ts', message['ts']),
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
        json.dump(ext_translations, open(f'data/translations/parsed_{output_name}.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)
        json.dump(translations, open(f'data/translations/only_text_{output_name}.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)


def handle_files():
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
    rus_without_eng_slang_corpus = get_corpus_diff(rus_slang_corpus, eng_corpus, 1000000)
    save_corpus(rus_without_eng_slang_corpus, 'data/corpus/translation-pure-slang.txt')


def main():
    # all_messages = read_data("data/kotlin_year")
    # groups_dict, unknown_msg, system_msg = group_threads(all_messages)
    # childs = []
    # for thread_ts, thread in groups_dict.items():
    #     childs.extend(map(lambda x: x['message'], thread['replies']))
    # translate_messages(childs, 'child')
    # list_a = json.load(open('data/translations/dont_touch_only_text_child.json', 'r', encoding='utf-8'))
    # list_b = json.load(open('data/translations/only_text_child.json', 'r', encoding='utf-8'))
    # list_r = list_a.extend(list_b)
    # json.dump(list_r, open('data/translations/result_only_text.json', 'w', encoding='utf-8'), ensure_ascii=False,
    #           indent=4)
    all_translated = read_data('data/translations/dont_touch_only_text_child.json')
    all_translated.extend(read_data('data/translations/only_text_topics.json'))
    lancaster_corpus = create_corpus(all_translated, LancasterStemmer(), stopwords.words('english'))
    porter_corpus = create_corpus(all_translated, PorterStemmer(), stopwords.words('english'))
    save_corpus(lancaster_corpus, 'data/corpus/lancaster_corpus.json')
    save_corpus(porter_corpus, 'data/corpus/porter_corpus.json')


if __name__ == "__main__":
    main()
