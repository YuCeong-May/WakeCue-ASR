"""Audio features and text-cue tokenisation."""

import json
import re

import torch
from torchaudio.compliance import kaldi


def normalize_cue(text):
    """Collapse whitespace; ``hicolmo`` spellings map to ``hi col mo`` (training convention)."""
    text = " ".join(str(text or "").strip().split())
    if "".join(text.casefold().split()) == "hicolmo":
        return "hi col mo"
    return text


def llm_cue(text):
    """Cue text as written into the LLM prompt: collapsed whitespace, lower case."""
    text = " ".join(str(text or "").split()).lower()
    if text.replace(" ", "") == "hicolmo":
        return "hi colmo"
    return text


class Fbank:
    """80-dim Kaldi fbank at 16 kHz (25 ms / 10 ms, povey window, no dither, no CMVN)."""

    def __init__(
        self,
        fs=16000,
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        window="povey",
    ):
        self.fs = fs
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.window = window

    def __call__(self, wav):
        """(N,) float waveform in [-1, 1] -> (1, T, 80) features and (1,) int32 lengths."""
        wav = torch.as_tensor(wav, dtype=torch.float32)
        num_samples = wav.shape[0]
        feats = kaldi.fbank(
            wav.unsqueeze(0),
            num_mel_bins=self.n_mels,
            frame_length=min(self.frame_length, num_samples / self.fs * 1000),
            frame_shift=self.frame_shift,
            dither=0.0,
            energy_floor=0.0,
            window_type=self.window,
            sample_frequency=self.fs,
            snip_edges=True,
        )
        lengths = torch.tensor([feats.size(0)], dtype=torch.int32)
        return feats[None].to(torch.float32), lengths


class CueTokenizer:
    """Cue text -> ids of the cue-encoder vocabulary, wrapped in ``<psok>`` ... ``<peok>``.

    Words are split on spaces and mapped through ``seg_dict`` (whole word first,
    then per character); anything unknown becomes ``<unk>``.
    """

    _WORD_CHAR = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")

    def __init__(self, token_list, seg_dict, unk="<unk>", psok="<psok>", peok="<peok>"):
        with open(token_list, encoding="utf-8") as f:
            tokens = json.load(f)
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.unk_id = self.token2id[unk]
        self.psok_id = self.token2id[psok]
        self.peok_id = self.token2id[peok]
        self.seg_dict = {}
        with open(seg_dict, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    self.seg_dict[parts[0]] = " ".join(parts[1:])

    def _segment(self, words):
        out = []
        for word in words:
            word = word.lower()
            if word in self.seg_dict:
                out.append(self.seg_dict[word])
            elif self._WORD_CHAR.match(word):
                out.extend(self.seg_dict.get(ch, "<unk>") for ch in word)
            else:
                out.append("<unk>")
        return " ".join(out).split()

    def encode(self, text):
        units = self._segment(normalize_cue(text).strip().split(" "))
        ids = [self.token2id.get(u, self.unk_id) for u in units]
        return [self.psok_id] + ids + [self.peok_id]
