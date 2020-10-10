from abc import ABC, abstractmethod
from datetime import datetime


class AbstractDbWriter(ABC):
    @abstractmethod
    def update_channels_states(self, channels_states: dict):
        """
        {
            "channel-id": {
                "name": str,
                "id": str,
                "following": bool,
                "last_update": timestamp(float)
            }
        }
        """
        pass

    @abstractmethod
    def get_channels_states(self) -> dict:
        pass

    @abstractmethod
    def add_messages(self, messages: list, channel_id: str):
        pass

    @abstractmethod
    def get_latest_timestamp(self, channel_id: str) -> float:
        pass
