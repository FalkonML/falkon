import os
from pathlib import Path
os.environ["NLTK_DATA"] = "/data/DATASETS/nltk_data"
os.environ["TORCH_HOME"] = "/data/giacomo/torch_home/"

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm


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
print(f"Loading Librispeech dataset {LIBRISPEECH_SPLIT}")
dataset = torchaudio.datasets.LIBRISPEECH(
    root=DATA_ROOT,
    url=LIBRISPEECH_SPLIT,
    download=True,
)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print(f"Loading torchaudio model {bundle._path}")
model = bundle.get_model().to(DEVICE)
model.eval()
print("Loaded Wav2Vec2 model")

@torch.no_grad()
def extract_features(waveform):
    waveform = waveform.to(DEVICE)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # Extract hidden representations
    features, _ = model.extract_features(waveform)
    # features is a list of hidden layers
    # Use final layer
    final_layer = features[-1]
    # Shape:
    # [1, T, D]
    return final_layer.squeeze(0).cpu()



for idx in tqdm(range(len(dataset))):
    waveform, sr, transcript, speaker_id, chapter_id, utt_id = dataset[idx]
    # Convert stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sr,
            SAMPLE_RATE,
        )
        waveform = resampler(waveform)
    try:
        # Frame-level embeddings
        frame_features = extract_features(waveform)
        # Utterance embedding
        utterance_embedding = frame_features.mean(dim=0)
        # Save
        save_dict = {
            "utt_id": utt_id,
            "transcript": transcript,
            "frame_features": frame_features,
            "utterance_embedding": utterance_embedding,
        }
        print(f"{save_dict=}")
        if idx > 5:
            break

        # save_path = os.path.join(
        #     FEATURE_ROOT,
        #     f"{utt_id}.pt"
        # )

        # torch.save(save_dict, save_path)
    except Exception as e:
        print(f"Failed on {utt_id}")
        print(e)

print("Done.")

# # ESPnet pretrained Conformer
# MODEL_NAME = "espnet/owsm_v3.1_ebf"
# print(f"Loading pretrained ESPnet model {MODEL_NAME}")
# d = ModelDownloader()
# model_data = d.download_and_unpack(MODEL_NAME)
# # speech2text = Speech2Text(
# #     asr_train_config=model["s2t_train_config"],
# #     asr_model_file=model["s2t_model_file"],
# #     device=DEVICE,
# # )
# model, train_args = ASRTask.build_model_from_file(
#     config_file=model_data["s2t_train_config"],
#     model_file=model_data["s2t_model_file"],
#     device=DEVICE,
# )
# # model = speech2text.asr_model
# model.eval()

# print("Model loaded.")


# # Collate function
# def collate_fn(batch):
#     waveforms = []
#     transcripts = []
#     utt_ids = []

#     for item in batch:
#         waveform, sr, transcript, speaker_id, chapter_id, utt_id = item

#         # Convert to mono if needed
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)

#         waveform = waveform.squeeze(0)

#         # Resample if needed
#         if sr != SAMPLE_RATE:
#             resampler = torchaudio.transforms.Resample(
#                 sr,
#                 SAMPLE_RATE,
#             )
#             waveform = resampler(waveform)

#         waveforms.append(waveform)
#         transcripts.append(transcript)
#         utt_ids.append(utt_id)

#     return waveforms, transcripts, utt_ids


# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# )

# # Feature extraction
# @torch.no_grad()
# def extract_encoder_features(waveform):
#     """
#     Extract frame-level encoder representations
#     from pretrained ESPnet Conformer.
#     """

#     waveform = waveform.to(DEVICE)

#     lengths = torch.tensor(
#         [waveform.shape[0]],
#         device=DEVICE,
#     )

#     waveform = waveform.unsqueeze(0)

#     # Frontend + encoder
#     feats, feats_len = model.encode(
#         speech=waveform,
#         speech_lengths=lengths,
#     )

#     # Shape:
#     # [1, T, D]
#     feats = feats.squeeze(0)

#     return feats.cpu()


# Main extraction loop
# print("Starting extraction...")
# i = 0
# for waveforms, transcripts, utt_ids in tqdm(loader):
#     waveform = waveforms[0]
#     transcript = transcripts[0]
#     utt_id = utt_ids[0]

#     try:
#         # Extract frame-level embeddings
#         frame_features = extract_encoder_features(waveform)

#         # Mean pooled utterance embedding
#         utterance_embedding = frame_features.mean(dim=0)

#         # Save
#         save_dict = {
#             "utt_id": utt_id,
#             "transcript": transcript,
#             "frame_features": frame_features,
#             "utterance_embedding": utterance_embedding,
#         }
#         print(save_dict)
#         if i > 5:
#             break

#         # save_path = os.path.join(
#         #     FEATURE_ROOT,
#         #     f"{utt_id}.pt",
#         # )

#         # torch.save(save_dict, save_path)

#     except Exception as e:
#         print(f"Failed on {utt_id}: {e}")
#     i += 1

# print("Done.")
