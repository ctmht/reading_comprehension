import os
from abc import ABC, abstractmethod
from typing import Any, Union


class GenericModel(ABC):
    """Interface for generic Llama7B model with full finetuning"""

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the base Llama7B model
        """
        super().__init__()
        # self.model = Load Llama7B here

    @abstractmethod
    def finetune(self, train_in, train_out, **kwargs) -> Any:
        """
        Function handling the finetuning procedure of the LLM
        on the given training data pairs in (train_in, train_out)
        """
        pass

    @abstractmethod
    def prompt(self, prompt: str, **kwargs) -> Any:
        """
        Function handling the generation of output
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, os.path], **kwargs) -> None:
        """
        Save pickled model at the specified path
        Or however we decide to save it...
        """
        # pickle(self.model)
        pass

    @abstractmethod
    def load(self, path: Union[str, os.path], **kwargs) -> None:
        """
        Save pickled model at the specified path
        """
        # self.model = pickle.(load from path)
        pass
