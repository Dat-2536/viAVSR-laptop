from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration import AVHubertAVSRConfig
from .nets.backend.e2e_asr_avhubert import E2E


@dataclass
class AVHubertAVSROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_ctc: Optional[torch.FloatTensor] = None
    loss_att: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None


class AVHubertAVSR(PreTrainedModel):
    """Released AV-HuBERT encoder with joint CTC/attention decoder."""

    config_class = AVHubertAVSRConfig

    def __init__(self, config: AVHubertAVSRConfig) -> None:
        super().__init__(config)
        self.avsr = E2E(config)

    def forward(
        self,
        videos: torch.Tensor,
        audios: torch.Tensor,
        labels: torch.Tensor,
        video_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> AVHubertAVSROutput:
        loss, loss_ctc, loss_att, acc = self.avsr(
            videos, audios, video_lengths, audio_lengths, labels
        )
        return AVHubertAVSROutput(
            loss=loss, loss_ctc=loss_ctc, loss_att=loss_att, acc=acc
        )
