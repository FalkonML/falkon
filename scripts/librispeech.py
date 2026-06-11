import os
from pathlib import Path
os.environ["NLTK_DATA"] = "/data/DATASETS/nltk_data"
os.environ["TORCH_HOME"] = "/data/giacomo/torch_home/"

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import ctc_segmentation

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/data/DATASETS/LibriSpeech/"
FEATURE_ROOT = "/data/DATASETS/LibriSpeech/features/"

LIBRISPEECH_SPLIT = "train-clean-360"

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
labels = bundle.get_labels()  # character-level vocabulary
print("Loaded Wav2Vec2 model")
print("labels:", labels)

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


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


def filter_word(w: str) -> bool:
    if w.endswith("'s"):
        return False
    if len(w) > 5:
        return False
    return True

def frame2word_embeddings(frame_features, segments, frame_rate):
    out = []
    for word, start, end in segments:
        s = int(start * frame_rate)
        e = int(end * frame_rate)
        emb = frame_features[s:e].mean(dim=0)
        out.append(emb)
    return out

all_word_features = []
all_words = []

for idx in tqdm(range(len(dataset))):
    waveform, sr, transcript, speaker_id, chapter_id, utt_id = dataset[idx]
    # stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(DEVICE)  # [batch-size, length in sample-rate]
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sr,
            SAMPLE_RATE,
        )
        waveform = resampler(waveform)
    # Frame-level embeddings
    with torch.no_grad():
        # [BS, T, D] features for the whole sample
        frame_features = extract_features(waveform)
        # emissions: [BS, T, Vocab] - emission probabilities for each character
        emissions, _ = model(waveform)
    # Soft segmentation with CTC.
    # See https://github.com/lumaku/ctc-segmentation for example code from
    # which this was derived (look at README 'Wav2Vec example code')
    words = transcript.split()
    log_probs = torch.nn.functional.log_softmax(emissions[0], dim=-1)
    log_probs_np = log_probs.numpy(force=True)
    config = ctc_segmentation.CtcSegmentationParameters(char_list=labels)
    config.index_duration = waveform.shape[1] / log_probs.shape[0] / SAMPLE_RATE
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, log_probs_np, ground_truth_mat)
    ctc_segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, words)
    # segments: [(word, start_time, end_time), ...]
    segments = [(w, p[0], p[1]) for w, p in zip(words, ctc_segments)] # word, start, end
    word_features = frame2word_embeddings(frame_features, segments, 1 / config.index_duration)
    lemm_words = []
    words = [w.lower() for w in words]
    tagged_words = pos_tag(words)
    for word, tag in tagged_words:
        lemm_words.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    include_words = [filter_word(w) for w in lemm_words]
    all_words.extend([lemm_words[i] for i in range(len(include_words)) if include_words[i]])
    all_word_features.extend([word_features[i] for i in range(len(include_words)) if include_words[i]])
    if idx % 10 == 0:
        print(f"{idx=} found {len(np.unique(all_words))} unique words out of {len(all_words)} total words")
        print(np.unique(all_words))

print("Done.")

