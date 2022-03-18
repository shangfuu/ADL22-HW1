from typing import List, Dict

import torch
import numpy as np
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        """Used for padding variable-length and convert to token_id.

        Args:
            samples (List[Dict]):   A batch of samples.
                                    [{'text':,'intent':,'id':},{},...]

        Returns:
            if labels in sampless
                Dict: {'features': padded token id, 'labels': intent to label id, 'id': list of string}
            else
                Dict: {'features': padded token id, 'labels': empty array, 'id': list of string}
        """
        # sample['text'] to padded token_id
        batch_tokens : List[List[str]] = [sample['text'].split() for sample in samples]
        padded_ids = self.vocab.encode_batch(batch_tokens=batch_tokens, to_len=self.max_len)

        if "intent" in samples[0]:
            labels = [self.label2idx(sample['intent']) for sample in samples]
        else:
            
            labels = []

        ids = [sample['id'] for sample in samples]

        return {'features': padded_ids, 'labels': labels, 'ids': ids}

        # TODO: implement collate_fn
        # raise NotImplementedError  
        
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SlotTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    
    @staticmethod
    def padd_label() -> int:
        return "O"

    def collate_fn(self, samples: List[Dict]) -> Dict:
        """Used for padding variable-length and convert to token_id.

        Args:
            samples (List[Dict]):   A batch of samples.
                                    [{'tokens':,'tags':,'id':},{},...]

        Returns:
            training:
                tokens: tags: ids:
            testing:
                tokens: length: ids:
        """
        
        batch_tokens : List[List[str]] = [sample['tokens'] for sample in samples]
        
        padded_ids = self.vocab.encode_batch(batch_tokens=batch_tokens, to_len=self.max_len)
        
        labels: List[List[int]] = []
        
        ids = [sample['id'] for sample in samples]

        if "tags" in samples[0]:
            # pad label
            for batch, sample in enumerate(samples):
                tmp = []
                # padding
                for idx, padded_id in enumerate(padded_ids[batch]):
                    if padded_id == 0:
                        # tag = self.label2idx(self.padd_label())
                        tag = self.num_classes
                    else:
                        tag = self.label2idx(sample['tags'][idx])
                    tmp.append(tag)
                labels.append(tmp)
            
            return {'tokens': padded_ids, 'tags': labels, 'ids': ids}
        else:
            length = [len(sample['tokens']) for batch, sample in enumerate(samples)]

            return {'tokens': padded_ids, 'length': length, 'ids': ids}

        # TODO: implement collate_fn
        # raise NotImplementedError
    
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
