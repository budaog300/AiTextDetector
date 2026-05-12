from typing import Dict
from src.model_service import BaseModelManager


class TextClassificationService:
    def __init__(self):
        self._managers = {}

    def initialize(self, name: str, manager: BaseModelManager):
        if name in self._managers:
            raise ValueError(f"Manager {name} already exists")
        self._managers[name] = manager

    def predict(self, manager_name: str, text: str, model_name: str):
        response = self._managers[manager_name].predict(text, model_name)
        return response

    def unload(self, manager_name: str):
        manager = self._managers.pop(manager_name, None)
        if manager:
            manager.unload()

    def get_available_managers(self):
        return list(self._managers.keys())

    def get_manager_models(self, manager_name: str):
        return list(self._managers[manager_name].models.keys())
