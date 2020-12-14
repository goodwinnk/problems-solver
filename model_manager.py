import logging

from nlp.message_processing import Message, read_data
from nlp.model import Model
from utils.abstractcontroller import AbstractController


class ModelManager:
    def __init__(self, models_path: str, controller: AbstractController):
        self.models_path = models_path
        self.controller = controller
        self.models_ = dict()

    def load_from_sources(self, msgs_path: str, dataset_path: str, channel_id: str):
        messages = read_data(msgs_path)
        dataset = list(read_data(dataset_path))
        new_channel_state = {channel_id: {
            "name": f"Name of {channel_id}",
            "id": channel_id,
            "following": True,
            "last_update": 0
        }}
        self.controller.remove_data(channel_id)
        self.controller.update_channels_states(new_channel_state)
        self.controller.add_parent_messages(messages, channel_id)
        self.controller.add_dataset_messages(channel_id, dataset)
        self.load_model(channel_id)

    def load_models(self, from_files=True):
        for channel in self.controller.get_channels_states().values():
            if channel['following']:
                self.models_[channel['id']] = self.load_model(channel['id'], load_from_file=from_files)

    def load_model(self, channel_id: str, load_from_file=True, create=True):
        if load_from_file:
            try:
                model = Model.load_model(f"{self.models_path}/{channel_id}")
                logging.info(f"Model for {channel_id} was founded in files")
                self.models_[channel_id] = model
                return model
            except (FileNotFoundError, TypeError):
                logging.warning(f'Cannot load/find model for {channel_id}.')
        if create:
            self.models_[channel_id] = self.create_model(channel_id)
            logging.info('Model was created from DB')
            return self.models_[channel_id]
        logging.warning("Get model return untrained model.")
        return Model()

    def get_model(self, channel_id: str) -> Model:
        if channel_id in self.models_:
            return self.models_[channel_id]
        return self.load_model(channel_id)

    def create_model(self, channel_id: str) -> Model:
        logging.info(f'Creating new model for channel: {channel_id}')
        messages = list(map(lambda m: Message.from_dict(m), self.controller.get_parent_messages(channel_id)))
        dataset = list(self.controller.get_dataset_data(channel_id))
        model = Model()
        model.train(messages, dataset)
        model.save_model(f"{self.models_path}/{channel_id}")
        logging.info('Model was created')
        return model

    def save_models(self):
        for channel_id, model in self.models_.items():
            model.save_model(f"{self.models_path}/{channel_id}")
