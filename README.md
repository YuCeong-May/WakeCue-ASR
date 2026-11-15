# WakeCue-ASR

![WakeCue-ASR architecture](assets/model.png)

![Training-data pipeline: (a) Qwen3-TTS synthesis from prompt speech, an ASR check with retry, and wake-word/command boundaries from Qwen3-ForcedAligner; (b) online simulation of target-speaker, interfering-speaker and noise mixtures with accept and reject labels](assets/data-pipeline.png)

**WakeCue-ASR** is a speech-LLM for enrollment-free, wake-word-guided
target-speaker ASR in noisy multi-talker scenes. Given the wake word as a text
cue, it transcribes the command of the speaker who said it, and returns
`<reject/>` when the cue is absent or the command comes from someone else. The
cue conditions a pretrained speech encoder through cross-attention, and a
LoRA-adapted Qwen2.5-0.5B decodes the transcript. The training data are
synthesized with Qwen3-TTS and mixed online with interfering speakers and noise.

- Paper: coming soon
- Model: [huggingface.co/YuCeong-May/WakeCue-ASR](https://huggingface.co/YuCeong-May/WakeCue-ASR)
- Demo: [WakeCue-ASR Space](https://huggingface.co/spaces/YuCeong-May/WakeCue-ASR)

## Inference

### Installation

Python 3.10 or later:

```bash
git clone https://github.com/YuCeong-May/WakeCue-ASR.git
cd WakeCue-ASR
pip install -r requirements.txt
```

For a specific CUDA build of PyTorch, install `torch` and `torchaudio` first
([pytorch.org](https://pytorch.org/get-started/locally/)). A GPU is optional
(about 1.5 s per 5 s clip on an Apple M3 CPU). The first run downloads
`model.pt` (1.6 GB, all weights including Qwen2.5-0.5B) from the
[model repository](https://huggingface.co/YuCeong-May/WakeCue-ASR); pass a
local copy with `--weights`.

### Command line

The input is Mandarin speech (resampled to 16 kHz mono; up to 20 s recommended)
and a text cue, usually the wake word (2-5 characters recommended). The demo's
examples are midea-test clips, each a wake word followed by a command:

```bash
hf download YuCeong-May/WakeCue-ASR --repo-type space --include "examples/*" --local-dir .
python infer.py --audio examples/midea_pos_2465.wav --cue 小美小美
```

Each input gives one JSON line:

```json
{"audio": "examples/midea_pos_2465.wav", "cue": "小美小美", "text": "小美小美播放一千零一夜故事", "rejected": false, "raw": "<psok>小美小美<peok>播放一千零一夜故事<|endoftext|>"}
```

`text` is the target speaker's transcript, including the cue. `rejected` is
`true` when there is no target speaker. `raw` is the decoder output:
`<psok>cue<peok>command`, `<psok>cue<peok><reject/>` when the command comes
from another speaker, or `<reject/>` when the cue is absent.

Separate wake-word and command recordings are joined in order, with no gap:

```bash
python infer.py --audio kws_5184.wav cmd_5184.wav --cue 小美小美
```

For many inputs, pass a JSON-lines list. `audio` is a path or a list of paths,
relative to the list file; other fields are copied to the output:

```
{"id": "pos_2465", "audio": "examples/midea_pos_2465.wav", "cue": "小美小美"}
{"id": "neg_5184", "audio": ["kws_5184.wav", "cmd_5184.wav"], "cue": "小美小美"}
```

```bash
python infer.py --list test.jsonl --output result.jsonl
```

`--device` defaults to `cuda` when it is available; `python infer.py --help`
lists all options.

### Python

From the repository root:

```python
from tcg_tsasr import TextCueGuidedTSASR

asr = TextCueGuidedTSASR.from_pretrained("YuCeong-May/WakeCue-ASR")  # or a local model.pt
asr.to("cuda")  # optional
out = asr.transcribe("examples/midea_pos_2465.wav", "小美小美")
print(out["text"], out["rejected"])
```

`transcribe` also accepts audio bytes or a `(waveform, sample_rate)` tuple.

## Results

### midea-test

midea-test, from the [Midea AI challenge](https://ai.midea.com/#/match-detail),
has 1838 smart-home wake-plus-command recordings with overlapping speech and
household noise (SNR -5 to 5 dB, at most two talkers at once). Values are in %:
pos-CER is the command CER on positive samples (a rejected positive counts as
deleting every character), FR and RR are the rejection rates on positive and
negative samples, and U = (100 - pos-CER + RR) / 2. B1-B6 are enrollment-free
baselines built from public pretrained models; the B2 threshold τ was tuned on
this test set. S1-S4 are the four training stages; the released model is S4.

| ID | System | pos-CER ↓ | FR ↓ | RR ↑ | U ↑ | Notes |
|---|---|---:|---:|---:|---:|---|
| B1 | Qwen3-ASR-1.7B | 32.21 | 0.00 | 0.00 | 33.89 | Direct ASR |
| B2 | CAM++ + Qwen3-ASR | 46.26 | 27.05 | 89.66 | 71.70 | Speaker gating & ASR; τ = 0.28 |
| B3 | Qwen2.5-Omni-7B | 82.08 | 28.08 | 27.00 | 22.46 | Audio-LLM prompting |
| B4 | MOSS-Transcribe-Diarize | 51.44 | 35.92 | 93.46 | 71.01 | Multi-talker ASR & text matching |
| B5 | WeSep + Qwen3-ASR | 73.99 | 1.25 | 29.11 | 27.56 | Speech separation & ASR |
| B6 | CocktailASR-1 | 29.19 | 9.60 | 66.67 | 68.74 | Wake segment as reference |
| S1 | WakeCue-ASR, Stage 1 (CTC) | 43.47 | 45.82 | 98.10 | 77.31 | Text-cue encoder |
| S2 | WakeCue-ASR, Stage 2 | 29.15 | 20.67 | 93.25 | 82.05 | LLM alignment |
| S3 | WakeCue-ASR, Stage 3 | 20.47 | 13.86 | 90.72 | 85.12 | Joint training |
| S4 | WakeCue-ASR, Stage 4 | **16.15** | **9.46** | 91.56 | **87.70** | Speaker supervision |

![pos-CER, RR and unified score U on midea-test for the six baselines and for WakeCue-ASR after Stage 1 and Stage 4](assets/results-midea-test.png)

### TS-Aishell

Two-speaker mixtures from AISHELL-1 and the AISHELL-2 test sets, 6000 per test
set. The cue is the first 2-5 characters of the target speaker's sentence.
CER / SER / reject rate in %:

| Test set | 2 chars | 3 chars | 4 chars | 5 chars |
|---|---|---|---|---|
| aishell1-ts | 4.21 / 20.32 / 0.92 | 2.69 / 17.08 / 0.35 | 2.09 / 15.23 / 0.32 | **1.83** / 13.90 / 0.57 |
| aishell-ios-ts | 11.56 / 44.55 / 3.67 | 8.77 / 38.73 / 3.12 | 7.22 / 34.32 / 4.47 | **6.34** / 29.25 / 7.58 |
| aishell-android-ts | 15.00 / 49.98 / 4.70 | 11.26 / 43.78 / 3.57 | 9.60 / 38.75 / 4.18 | **8.76** / 34.58 / 6.50 |
| aishell-mic-ts | 12.21 / 46.02 / 3.85 | 9.33 / 39.63 / 3.23 | 7.62 / 34.93 / 4.10 | **6.42** / 30.23 / 6.68 |
| aishell-mix-ts | 8.35 / 37.55 / 2.78 | 5.91 / 32.37 / 2.63 | 4.76 / 28.55 / 3.32 | **3.94** / 23.70 / 5.40 |

## License

[Apache-2.0](LICENSE)
