from abc import ABC, abstractmethod


class AbstractDbWriter(ABC):
    @abstractmethod
    def update_channels_states(self, channels_states: dict):
        """
        {
            "channel_id": {
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
    def add_parent_messages(self, messages: list, channel_id: str):
        pass

    @abstractmethod
    def add_child_messages(self, messages: list, channel_id: str, parent_ts: str):
        pass

    @abstractmethod
    def get_latest_timestamp(self, channel_id: str) -> float:
        pass
