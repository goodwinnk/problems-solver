import json
from pprint import pprint

from nlp.message_processing import read_data, group_threads


def normalize_topics():
    messages = read_data('data/kotlin_year')
    topics = read_data('data/processed/old_all_topics.json')
    result, system = group_threads(messages)
    new_topics = []
    for key, value in result.items():
        channel_id = value.get('channel_id', 'UNKNOWN')
        for index, texts, in enumerate(map(lambda x: (x['origin'], x['text']), topics)):
            if value['text'] == texts[0]:  # origin
                new_topics.append({
                    'text': texts[1],  # text = origin if is_eng else translate(origin)
                    'ts': value['ts'],
                    'channel_id': channel_id,
                    'id': index
                })
                break
    json.dump(new_topics, open('data/processed/all_topics.json', 'w'), ensure_ascii=False, indent=4)


def normalize_dataset(path):
    data = read_data(path)
    topics = read_data('data/processed/щдв_all_topics.json')
    topics = dict(map(lambda x: (str(x['id']), x), topics))
    normalized = dict()
    for key, value in data.items():
        message = topics[key]
        new_key = f'{message["channel_id"]}-{message["ts"]}'
        normalized[new_key] = {
            'origin': message['text'],
            'channel_id': message['channel_id'],
            'ts': message['ts'],
            'similars': []
        }
        for item in value['similars']:
            child_message = topics[str(item['id'])]
            normalized[new_key]['similars'].append({
                'ts': child_message['ts'],
                'channel_id': child_message['channel_id'],
                'text': child_message['text']
            })
    json.dump(normalized, open(f'{path}_edited.json', 'w'), ensure_ascii=False, indent=4)


def only_positives():
    data = read_data('data/dataset/copy/code_0.2.json')
    topics = read_data('data/processed/all_topics.json')
    topics = dict(map(lambda x: (str(x['id']), x), topics))
    positives = set()
    for key, value in data.items():
        message = topics[key]
        parent_key = f'{message["channel_id"]}-{message["ts"]}'
        for item in value['similars']:
            child_message = topics[str(item['id'])]
            child_key = child_message['channel_id'] + '-' + child_message['ts']
            positives.add((parent_key, child_key, True))
            positives.add((child_key, parent_key, True))
    json.dump(list(positives), open(f'code_positives.json', 'w', ), ensure_ascii=False, indent=4)


def main():
    # only_positives()
    normalize_dataset('data/dataset/tfidf+svd_0.7.json')


if __name__ == '__main__':
    main()
