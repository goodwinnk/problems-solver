import json
from nlp.message_processing import Message, read_data
from nlp.dual_model import DualModel
from random import choices, shuffle

if __name__ == '__main__':
    positives = read_data('data/dataset/positives/all_positives.json')
    dataset = read_data('data/dataset/TRY_IT.json')
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    model = DualModel()
    model.train(messages, dataset)
    model.test(messages, dataset, do_train=False)
    print(len(messages), len(positives))
    negatives = []
    approx_negatives_count = len(dataset)
    shuffle(messages)
    for message in messages:
        result = model.get_similar_messages(message)
        for sim_message in result:
            if sim_message != message:
                record = [sim_message.get_key(), message.get_key(), True]
                if record not in positives:
                    negatives.append([record[0], record[1], False])
        if len(negatives) >= approx_negatives_count:
            break
    print(f'Actually, negatives: {len(negatives)}')
    new_dataset = dataset.copy()
    new_dataset.extend(negatives)

    model = DualModel()
    model.train(messages, new_dataset)
    model.test(messages, new_dataset, do_train=False)
    json.dump(new_dataset, open('data/dataset/TRY_IT.json', 'w'), indent=4)
