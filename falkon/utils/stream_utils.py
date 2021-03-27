from typing import Optional

import torch


def sync_current_stream(device: Optional[torch.device] = None) -> None:
    # Sync current stream
    stream = torch.cuda.current_stream(device)
    stream.synchronize()


def get_non_default_stream(device: Optional[torch.device] = None) -> torch.cuda.Stream:
    # Chooses the current stream if it's not the default stream.
    # If the current stream is the default stream, creates a new stream.
    stream = torch.cuda.current_stream(device)
    # noinspection PyProtectedMember
    if stream._as_parameter_ == torch.cuda.default_stream(device)._as_parameter_:
        stream = torch.cuda.Stream(device)
    return stream
