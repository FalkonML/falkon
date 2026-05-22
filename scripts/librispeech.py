import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/data/DATASETS/LibriSpeech/"
FEATURE_ROOT = "/data/DATASETS/LibriSpeech/features/"

LIBRISPEECH_SPLIT = "train-clean-100"

BATCH_SIZE = 1
SAMPLE_RATE = 16000


# Create output directory
Path(FEATURE_ROOT).mkdir(parents=True, exist_ok=True)


# Load LibriSpeech
dataset = torchaudio.datasets.LIBRISPEECH(
    root=DATA_ROOT,
    url=LIBRISPEECH_SPLIT,
    download=True,
)

# ESPnet pretrained Conformer
print("Loading pretrained ESPnet model...")

d = ModelDownloader()

speech2text = Speech2Text(
    **d.download_and_unpack(
        "espnet/owsm_v3.1_ebf"
    ),
    device=DEVICE,
)

model = speech2text.asr_model
model.eval()

print("Model loaded.")


# Collate function
def collate_fn(batch):
    waveforms = []
    transcripts = []
    utt_ids = []

    for item in batch:
        waveform, sr, transcript, speaker_id, chapter_id, utt_id = item

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sr,
                SAMPLE_RATE,
            )
            waveform = resampler(waveform)

        waveforms.append(waveform)
        transcripts.append(transcript)
        utt_ids.append(utt_id)

    return waveforms, transcripts, utt_ids


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

# Feature extraction
@torch.no_grad()
def extract_encoder_features(waveform):
    """
    Extract frame-level encoder representations
    from pretrained ESPnet Conformer.
    """

    waveform = waveform.to(DEVICE)

    lengths = torch.tensor(
        [waveform.shape[0]],
        device=DEVICE,
    )

    waveform = waveform.unsqueeze(0)

    # Frontend + encoder
    enc, enc_len = model.encode(
        speech=waveform,
        speech_lengths=lengths,
    )

    # Shape:
    # [1, T, D]
    enc = enc.squeeze(0)

    return enc.cpu()


# Main extraction loop
print("Starting extraction...")
i = 0
for waveforms, transcripts, utt_ids in tqdm(loader):
    waveform = waveforms[0]
    transcript = transcripts[0]
    utt_id = utt_ids[0]

    try:
        # Extract frame-level embeddings
        frame_features = extract_encoder_features(waveform)

        # Mean pooled utterance embedding
        utterance_embedding = frame_features.mean(dim=0)

        # Save
        save_dict = {
            "utt_id": utt_id,
            "transcript": transcript,
            "frame_features": frame_features,
            "utterance_embedding": utterance_embedding,
        }
        print(save_dict)
        if i > 5:
            break

        # save_path = os.path.join(
        #     FEATURE_ROOT,
        #     f"{utt_id}.pt",
        # )

        # torch.save(save_dict, save_path)

    except Exception as e:
        print(f"Failed on {utt_id}: {e}")
    i += 1

print("Done.")