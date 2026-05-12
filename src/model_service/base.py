from abc import ABC, abstractmethod


class BaseModelManager(ABC):
    @abstractmethod
    def load(self) -> dict: ...

    def predict(self, text: str, model_name: str) -> dict: ...

    def unload(self): ...
