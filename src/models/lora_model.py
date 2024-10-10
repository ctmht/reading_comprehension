from typing import Any, Union
import os

from .generic_llama7b import GenericLlama7B


class LoRAModel(GenericLlama7B):
	""" Llama7B finetuned using LoRA """
	
	def __init__(
		self,
		**kwargs
	):
		"""
		Initialize the base Llama7B model
		"""
		# Load Llama7B into self.model
		super().__init__()
	
	
	def finetune(
		self,
		train_in,
		train_out,
		lora_rank: int = 16,
		**kwargs
	) -> Any:
		"""
		Finetuning procedure for the LLM using Low Rank Adaptation (LoRA)
		on the given training data pairs in (train_in, train_out)
		
		Args:
			train_in: the input data
			train_out: the output data
			lora_rank: rank to use in LoRA procedure
			...
		Returns:
			...
		Raises:
			...
		"""
		pass
	
	
	def prompt(
		self,
		prompt: str,
		**kwargs
	) -> Any:
		"""
		Function handling the generation of output
		"""
		pass
	
	
	def save(
		self,
		path: Union[str, os.path],
		**kwargs
	) -> None:
		"""
		Save pickled model at the specified path
		"""
		# pickle(self.model)
		pass
	
	
	def load(
		self,
		path: Union[str, os.path],
		**kwargs
	) -> None:
		"""
		Save pickled model at the specified path
		"""
		# self.model = pickle.(load from path)
		pass