import logging
from random import shuffle

from nlp.message_processing import Message, read_data
from nlp.model import Model
from utils.abstractcontroller import AbstractController


class ModelManager:
    def __init__(self, models_path: str, controller: AbstractController):
        self.models_path = models_path
        self.controller = controller
        self.models_ = dict()

    def load_from_sources(self, msgs_path: str, dataset_path: str, channel_id: str):
        messages = [Message.from_dict(msg) for msg in read_data(msgs_path)]
        dataset = read_data(dataset_path)
        shuffle(dataset)
        model = Model()
        model.train(messages, dataset)
        model.save_model(f"{self.models_path}/{channel_id}")

    def load_models(self):
        for channel in self.controller.get_channels_states().values():
            if channel['following']:
                self.models_[channel['id']] = self.get_model(channel['id'])

    def get_model(self, channel_id: str) -> Model:
        if channel_id in self.models_:
            return self.models_[channel_id]
        try:
            model = Model.load_model(f"{self.models_path}/{channel_id}")
            logging.info(f"Model for {channel_id} was founded in files")
            return model
        except (FileNotFoundError, TypeError):
            logging.warning(f'Cannot load/find model for {channel_id}.')
        return self.create_model(channel_id)

    def create_model(self, channel_id: str) -> Model:
        logging.info(f'Creating new model for channel: {channel_id}')
        messages = list(map(lambda m: Message.from_dict(m), self.controller.get_parent_messages(channel_id)))
        dataset = list(map(lambda m: Message.from_dict(m), self.controller.get_dataset_messages(channel_id)))
        model = Model()
        model.train(messages, dataset)
        model.save_model(f"{self.models_path}/{channel_id}")
        logging.info('Model was created')
        return model
