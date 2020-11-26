import json
from nlp.message_processing import Message, read_data
from nlp.dual_model import DualModel, DummyClassifier
from random import choices, shuffle

if __name__ == '__main__':
    messages = [Message.from_dict(msg) for msg in read_data('data/processed/all_topics.json')]
    model = DualModel()
    dataset = []
    shuffle(dataset)
    model.classifier = DummyClassifier()
    model.train(messages, dataset)
    positives = read_data('data/dataset/positives/all_positives.json')
    total, unknown_counter = 0, 0
    negatives = []
    for message in messages:
        result, similarities = model.get_similar_messages(message)
        total += len(result)
        for sim_message, sim in zip(result, similarities):
            if sim_message != message:
                record = [sim_message.get_key(), message.get_key(), True]
                if record not in positives:
                    if [record[0], record[1], False] not in negatives and \
                            [record[1], record[0], False] not in negatives:
                        if sim[0] >= 0.5 or sim[1] >= 0.2:
                            negatives.append([record[0], record[1], False])
                        elif choices(range(5), k=1)[0] == 0:
                            negatives.append([record[0], record[1], False])
    print(f'Negatives: {len(negatives)}')
    dataset = positives.copy()
    new_dataset = dataset.copy()
    new_dataset.extend(negatives)

    model = DualModel()
    model.train(messages, new_dataset)
    model.test(messages, new_dataset, do_train=False)
    json.dump(new_dataset, open('data/dataset/TRY_IT.json', 'w'), indent=4)
