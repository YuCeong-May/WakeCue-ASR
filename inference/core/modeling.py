"""Text-Cue-Guided TS-ASR model (inference only).

Given a mixture recording and a short text cue spoken by the target speaker, the
model transcribes only that speaker::

    cue tokens -> CueEncoder -----------------.
                                              v (cross-attention in every layer)
    fbank -----> AcousticEncoder (Conformer) -> AudioProjector -> Qwen2.5-0.5B + LoRA

The LLM input is ``<|endoftext|> prompt(cue) <audio> audio_tokens </audio>`` and
the output is ``<psok>cue<peok>transcript``, or ends in ``<reject/>`` when the
cue speaker is absent. Attribute names follow the training code so ``model.pt``
loads with ``strict=True``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import AcousticEncoder, CueEncoder
from .frontend import llm_cue

# Instruction the model was trained with; it must stay verbatim (in Chinese).
# Roughly: "Task: target-speaker ASR. Cue: {cue}. Transcribe only the person who
# says the cue and ignore other speakers. Output <psok>cue<peok>text; if the text
# is not from the cue speaker output <psok>cue<peok><reject/>; if the cue is not
# in the audio output <reject/>. Audio:"
PROMPT = (
    "任务：目标说话人语音识别。线索词：{cue}。"
    "只转录说出该线索词那个人的语音，忽略其他说话人。"
    "输出格式为 <psok>线索词<peok>目标文本；"
    "若目标文本不是线索词说话人所说，则输出 <psok>线索词<peok><reject/>；"
    "若线索词未出现在音频中，则输出 <reject/>。音频如下："
)

# Added to the Qwen vocabulary in this exact order (ids 151665-151671); the
# embedding and lm_head rows in the checkpoint depend on it.
SPECIAL_TOKENS = [
    "<audio>",
    "</audio>",
    "<|audio_pad|>",
    "<|asr|>",
    "<psok>",
    "<peok>",
    "<reject/>",
]

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]

# Training-only auxiliary speaker-classification head stored in the checkpoint.
_TRAINING_ONLY_PREFIXES = ("spk_head.",)


class AudioProjector(nn.Module):
    """Encoder states (B, T, D_enc) -> LLM embeddings (B, ceil(T / stride), D_llm).

    A gated 4-frame convolution with a residual over the stacked window, then a
    bottleneck and an output LayerNorm. At stride 2 the 25 Hz encoder becomes
    12.5 audio tokens per second.
    """

    def __init__(self, encoder_dim=256, llm_dim=896, stride=2, kernel_size=4):
        super().__init__()
        hidden_dim = kernel_size * encoder_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.gate = nn.Conv1d(
            encoder_dim, hidden_dim, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.up = nn.Conv1d(
            encoder_dim, hidden_dim, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_bottleneck = nn.Linear(hidden_dim, encoder_dim)
        self.to_llm = nn.Linear(encoder_dim, llm_dim)
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, x, lengths):
        batch, time, dim = x.shape
        valid = torch.arange(time, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        x = x * valid.unsqueeze(-1).to(x.dtype)
        out_time = (time + self.stride - 1) // self.stride
        pad = self.stride * out_time + (self.kernel_size - self.stride) - time
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        x_conv = x.transpose(1, 2)
        hidden = F.silu(self.gate(x_conv).transpose(1, 2)) * self.up(x_conv).transpose(1, 2)
        hidden = self.hidden(hidden)
        residual = (
            x.unfold(1, self.kernel_size, self.stride)
            .permute(0, 1, 3, 2)
            .reshape(batch, out_time, self.kernel_size * dim)
        )
        hidden = self.norm(hidden + residual)
        hidden = self.out_norm(self.to_llm(self.to_bottleneck(hidden)))
        out_lengths = (lengths.to(torch.long) + self.stride - 1) // self.stride
        return hidden, out_lengths.clamp_max(hidden.size(1))


class TextCueGuidedTSASRModel(nn.Module):
    """Text-Cue-Guided target-speaker ASR."""

    def __init__(self, qwen_dir, cue_vocab_size=8406, projector_stride=2, lora_rank=64):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        self.cue_encoder = CueEncoder(num_token=cue_vocab_size)
        self.encoder = AcousticEncoder()

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_dir, local_files_only=True)
        self.qwen_tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        config = Qwen2Config.from_pretrained(qwen_dir, local_files_only=True)
        config.vocab_size = len(self.qwen_tokenizer)
        # Every Qwen weight (base and LoRA) is in model.pt, so the base
        # checkpoint is not needed: build from the config and load model.pt.
        qwen = Qwen2ForCausalLM._from_config(config, dtype=torch.bfloat16)
        self.qwen = get_peft_model(
            qwen,
            LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=True,
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.05,
                target_modules=LORA_TARGETS,
            ),
        )
        # LoRA adapters were trained (and saved) in fp32 on top of the bf16 base.
        for name, param in self.qwen.named_parameters():
            if "lora_" in name:
                param.data = param.data.float()
        self.qwen_dtype = torch.bfloat16

        self.projector = AudioProjector(
            encoder_dim=self.encoder.output_dim,
            llm_dim=config.hidden_size,
            stride=projector_stride,
        )

        tok = self.qwen_tokenizer
        self.bos_id = int(tok.bos_token_id if tok.bos_token_id is not None else config.bos_token_id)
        self.eos_id = int(tok.eos_token_id)
        ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
        self.audio_id = ids["<audio>"]
        self.audio_end_id = ids["</audio>"]
        self.audio_pad_id = ids["<|audio_pad|>"]
        self.reject_id = ids["<reject/>"]
        pre, post = PROMPT.split("{cue}")
        self.prompt_pre_ids = tok.encode(pre, add_special_tokens=False)
        self.prompt_post_ids = tok.encode(post, add_special_tokens=False)

    def load_checkpoint(self, path):
        """Load ``model.pt`` strictly; returns the number of tensors loaded."""
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = state.get("state_dict", state)
        state = {k: v for k, v in state.items() if not k.startswith(_TRAINING_ONLY_PREFIXES)}
        self.load_state_dict(state, strict=True)
        return len(state)

    def _cue_ids(self, cue):
        text = llm_cue(cue)
        ids = self.qwen_tokenizer.encode(text, add_special_tokens=False) if text else []
        return ids or [self.audio_pad_id]

    @torch.no_grad()
    def transcribe(self, speech, speech_lengths, cue_tokens, cue_text, max_new_tokens=256):
        """Greedy-decode one utterance.

        Args:
            speech: (1, T, 80) fbank.
            speech_lengths: (1,) frame count.
            cue_tokens: cue-encoder ids including ``<psok>`` / ``<peok>``.
            cue_text: cue as typed; goes into the LLM prompt.

        Returns:
            dict with ``text`` (special tokens removed), ``text_raw`` and ``reject``.
        """
        device = speech.device
        cue = torch.tensor([cue_tokens], dtype=torch.long, device=device)
        cue_len = torch.tensor([len(cue_tokens)], dtype=torch.int32, device=device)
        cue_feat, cue_valid = self.cue_encoder(cue, cue_len)
        enc, enc_lens = self.encoder(
            speech, speech_lengths.to(device), cue_feat.to(speech.dtype), cue_valid
        )
        audio, audio_lens = self.projector(enc, enc_lens)
        audio = audio[:, : int(audio_lens[0])]

        head = (
            [self.bos_id]
            + self.prompt_pre_ids
            + self._cue_ids(cue_text)
            + self.prompt_post_ids
            + [self.audio_id]
        )
        embed = self.qwen.get_input_embeddings()
        head = embed(torch.tensor([head], dtype=torch.long, device=device))
        tail = embed(torch.tensor([[self.audio_end_id]], dtype=torch.long, device=device))
        inputs = torch.cat([head, audio, tail], dim=1)
        mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=device)
        out = self.qwen.generate(
            inputs_embeds=inputs.to(self.qwen_dtype),
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
            use_cache=True,
        )
        ids = [int(x) for x in out[0].tolist()]
        return {
            "text": self.qwen_tokenizer.decode(ids, skip_special_tokens=True),
            "text_raw": self.qwen_tokenizer.decode(ids, skip_special_tokens=False),
            "reject": self.reject_id in ids,
        }
