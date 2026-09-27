import math

import torch
from torch import nn

from .mem_modules import KeyProjection, MemoryDecoder, ValueEncoder


def get_similarity(memory_key, memory_shrinkage, query_key, query_selection):
    key_dim = memory_key.shape[1]
    memory_key = memory_key.flatten(start_dim=2)
    query_key = query_key.flatten(start_dim=2)
    if memory_shrinkage is not None:
        memory_shrinkage = memory_shrinkage.flatten(start_dim=1).unsqueeze(2)
    if query_selection is not None:
        query_selection = query_selection.flatten(start_dim=2)
        memory_key = memory_key.transpose(1, 2)
        similarity = (
            -(memory_key.square() @ query_selection)
            + 2.0 * (memory_key @ (query_key * query_selection))
            - (query_selection * query_key.square()).sum(1, keepdim=True)
        )
    else:
        similarity = (
            -memory_key.square().sum(1).unsqueeze(2)
            + 2.0 * (memory_key.transpose(1, 2) @ query_key)
        )
    similarity = similarity / math.sqrt(key_dim)
    if memory_shrinkage is not None:
        similarity = similarity * memory_shrinkage
    return similarity


def get_affinity(memory_key, memory_shrinkage, query_key, query_selection):
    similarity = get_similarity(
        memory_key,
        memory_shrinkage,
        query_key,
        query_selection,
    )
    return torch.softmax(similarity, dim=1)


def readout(affinity, memory_value):
    batch_size, channels, frames, height, width = memory_value.shape
    memory_value = memory_value.view(batch_size, channels, frames * height * width)
    memory = torch.bmm(memory_value, affinity)
    return memory.view(batch_size, channels, height, width)


class Mem(nn.Module):
    """Feature memory used to propagate the first-frame prompt through a video."""

    def __init__(self, config):
        super().__init__()
        self.key_dim = int(config.get("key_dim", 64))
        self.value_dim = int(config.get("value_dim", 256))
        self.hidden_dim = int(config.get("hidden_dim", 64))
        self.single_object = bool(config.get("single_object", False))

        self.value_encoder = ValueEncoder(
            self.value_dim,
            self.hidden_dim,
            self.single_object,
        )
        self.key_proj = KeyProjection(256, self.key_dim)
        self.decoder = MemoryDecoder(self.value_dim, self.hidden_dim)

    def encode_value(self, frame, embedding, hidden, masks, is_deep_update=True):
        other_masks = torch.zeros_like(masks)
        value, hidden = self.value_encoder(
            frame,
            embedding,
            hidden,
            masks,
            other_masks,
            is_deep_update,
        )
        return value.unsqueeze(3), hidden

    def read_memory(
        self,
        query_key,
        query_selection,
        memory_key,
        memory_shrinkage,
        memory_value,
    ):
        batch_size, num_objects = memory_value.shape[:2]
        flat_value = memory_value.flatten(start_dim=1, end_dim=2)
        affinity = get_affinity(
            memory_key,
            memory_shrinkage,
            query_key,
            query_selection,
        )
        memory = readout(affinity, flat_value)
        return memory.view(
            batch_size,
            num_objects,
            self.value_dim,
            *memory.shape[-2:],
        )

    def decode(self, embedding, hidden, memory_readout):
        return self.decoder(embedding, hidden, memory_readout)

    def forward(self, mode, *args, **kwargs):
        if mode == "encode_value":
            return self.encode_value(*args, **kwargs)
        if mode == "read_memory":
            return self.read_memory(*args, **kwargs)
        if mode == "decode":
            return self.decode(*args, **kwargs)
        raise ValueError(f"Unsupported memory operation: {mode}")
