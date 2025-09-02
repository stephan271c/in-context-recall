import torch
from .new_mem_module import metaRNN
from .synthetic_datasets import InContextRecallDataset

def evaluate(dataset: InContextRecallDataset, model: metaRNN, num_steps: int = 100):