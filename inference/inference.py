"""High-level inference: audio + text cue -> transcript of the cue speaker.

    from inference import WakeCueASR

    asr = WakeCueASR.from_pretrained("YuCeong-May/WakeCue-ASR")
    asr.transcribe("speech.wav", "附近居民")
"""

import io
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .core.frontend import CueTokenizer, Fbank, normalize_cue
from .core.modeling import WakeCueASRModel

ASSETS = Path(__file__).resolve().parent / "assets"
SAMPLE_RATE = 16000


def load_audio(source):
    """Path, bytes or ``(waveform, sample_rate)`` -> mono float32 waveform at 16 kHz."""
    if isinstance(source, tuple):
        wav, sr = source
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
    else:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        try:
            wav, sr = sf.read(source, dtype="float32", always_2d=True)
        except (RuntimeError, sf.LibsndfileError) as exc:
            raise ValueError("Could not read the audio. Please upload a valid audio file.") from exc
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(sr, SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype("float32")
    return wav


class WakeCueASR:
    """Audio decoding, features, cue tokenisation and greedy decoding around the model."""

    def __init__(self, model):
        self.model = model
        self.num_tensors = None
        self.device = "cpu"
        self.fbank = Fbank(fs=SAMPLE_RATE)
        self.cue_tokenizer = CueTokenizer(ASSETS / "cue_vocab.json", ASSETS / "cue_seg_dict.txt")

    @classmethod
    def from_pretrained(cls, path_or_repo, **hub_kwargs):
        """Load ``model.pt`` from a local file or directory, or from a Hugging Face model repo."""
        path = Path(path_or_repo)
        if path.is_dir():
            path = path / "model.pt"
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(str(path_or_repo), "model.pt", **hub_kwargs)
        model = WakeCueASRModel(ASSETS / "qwen")
        num_tensors = model.load_checkpoint(path)
        model.eval()
        asr = cls(model)
        asr.num_tensors = num_tensors
        return asr

    def to(self, device):
        if self.device != device:
            self.model.to(device)
            self.device = device
        return self

    def transcribe(self, audio, cue, max_new_tokens=256):
        """Transcribe the speaker who says ``cue``.

        Returns a dict with ``text`` (transcript), ``raw`` (decoder output with
        special tokens), ``rejected`` and the normalised ``cue``.
        """
        cue = normalize_cue(cue)
        if not cue:
            raise ValueError("The text cue is empty.")
        wav = load_audio(audio)
        if not np.isfinite(wav).all() or np.max(np.abs(wav), initial=0.0) < 1e-7:
            raise ValueError("The audio is empty, silent or contains invalid samples.")

        feats, lengths = self.fbank(wav)
        result = self.model.transcribe(
            feats.to(self.device),
            lengths,
            self.cue_tokenizer.encode(cue),
            cue,
            max_new_tokens=max_new_tokens,
        )
        raw = result["text_raw"]
        text = re.sub(r"<[^>]*>", "", raw).replace("@@", "").strip()
        return {
            "text": text,
            "raw": raw,
            "rejected": result["reject"] or not text,
            "cue": cue,
        }
