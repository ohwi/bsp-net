# based on https://github.com/rwightman/pytorch-image-models/blob/main/timm/data/loader.py
from contextlib import suppress
from functools import partial

import torch
import torch.utils.data


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    batch_size = len(batch)
    new_size = (batch_size, *batch[0].shape)
    new_batch = torch.zeros(new_size)

    for i in range(batch_size):
        new_batch[i] = torch.from_numpy(batch[i])

    return new_batch


class PrefetchLoader:
    def __init__(
            self,
            loader,
            device=torch.device('cuda'),
):
        self.loader = loader
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_batch in self.loader:

            with stream_context():

                next_batch = next_batch.to(device=self.device, non_blocking=True)

            if not first:
                yield batch
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            batch = next_batch

        yield batch

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
