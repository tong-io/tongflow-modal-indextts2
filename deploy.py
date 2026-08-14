"""Modal deploy entry for indextts2 (IndexTTS-2.5).

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal
from tongflow import deploy


# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = ["text-audio-gen-speech"]

_cfg: dict[str, Any] = {}


def _repo_id() -> str:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and repos:
        r = repos[0]
        if isinstance(r, dict) and r.get("repoId"):
            return str(r["repoId"])
    return "IndexTeam/IndexTTS-2.5"


REPO_ID = _repo_id()
MODEL_DIR = f"/models/{REPO_ID}"

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.text_audio_gen_speech import (
    TextAudioGenSpeechInput,
    TextAudioGenSpeechOutput,
)
from tongflow.models.text_gen_speech_clone import (
    TextGenSpeechCloneInput,
    TextGenSpeechCloneOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import node_slot


app = modal.App(Path(__file__).resolve().parent.name)

INDEXTTS_GIT_REF = "v2.5.0"
# Upstream recommends emo_alpha around 0.6 for natural emotion-text guidance.
EMO_ALPHA = 0.6
# Languages with first-class IndexTTS-2.5 support; anything else falls back to
# the tokenizer's shared "common" language token.
LANG_BY_NAME = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "spanish": "es",
    "arabic": "ar",
}


def _lang(language: str | None) -> str:
    v = (language or "").strip().lower()
    if not v or v == "auto":
        return "zh"
    return LANG_BY_NAME.get(v, v)


image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "tongflow==0.2.21",
        "fastapi[standard]",
        "soundfile==0.13.1",
        f"indextts @ git+https://github.com/index-tts/index-tts.git@{INDEXTTS_GIT_REF}",
    )
    # indextts' dependency chain resolves protobuf down to 3.x, which breaks
    # the Modal client runtime baked into the container. Re-pin it last.
    .pip_install("protobuf==5.29.5")
)

with image.imports():
    import io
    import soundfile as sf
    from indextts.infer_v2_5 import IndexTTS2


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="L4",
    volumes={"/models": volume},
)
class Speech:
    @modal.enter()
    def load(self):
        # use_qwen_emo loads the bundled QwenEmotion text-to-emotion model,
        # required for emo_text guidance on the text-audio-gen-speech slot.
        self.tts = IndexTTS2(
            cfg_path=f"{MODEL_DIR}/config.yaml",
            model_dir=MODEL_DIR,
            use_bf16=True,
            use_qwen_emo=True,
        )

    def _synthesize(
        self,
        text: str,
        spk_audio_path: str,
        lang: str,
        emo_text: str | None = None,
    ) -> bytes:
        kwargs: dict[str, Any] = {}
        if emo_text:
            kwargs.update(
                use_emo_text=True, emo_text=emo_text, emo_alpha=EMO_ALPHA
            )
        result = self.tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
            lang=lang,
            **kwargs,
        )
        if result is None:
            raise RuntimeError("IndexTTS returned no audio")
        sr, wav = result
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        text: str,
        spk_audio_path: str,
        lang: str = "zh",
        emo_text: str = "",
    ) -> bytes:
        return self._synthesize(
            text=text,
            spk_audio_path=spk_audio_path,
            lang=lang,
            emo_text=emo_text or None,
        )

    @modal.method()
    @node_slot(NodeSlots.TEXT_GEN_SPEECH_CLONE)
    def text_gen_speech_clone(
        self,
        input: TextGenSpeechCloneInput,
    ) -> TextGenSpeechCloneOutput:
        text = (input.text or "").strip()
        if not text or input.ref_audio is None:
            return TextGenSpeechCloneOutput(
                success=False, error="Missing text or ref_audio"
            )
        # IndexTTS clones directly from the reference waveform; ref_text /
        # x_vector_only / max_new_tokens are Qwen-style knobs with no
        # counterpart here and are intentionally ignored.
        with asset_as_path(input.ref_audio) as ref_path:
            raw = self._synthesize(
                text=text,
                spk_audio_path=str(ref_path),
                lang=_lang(input.language),
            )
        return TextGenSpeechCloneOutput(
            success=True, audio=asset(raw, mime="audio/wav")
        )

    @modal.method()
    @node_slot(NodeSlots.TEXT_AUDIO_GEN_SPEECH)
    def text_audio_gen_speech(
        self,
        input: TextAudioGenSpeechInput,
    ) -> TextAudioGenSpeechOutput:
        text = (input.text or "").strip()
        if not text or input.audio is None:
            return TextAudioGenSpeechOutput(
                success=False, error="Missing text or audio"
            )
        emotion = (input.emotion or "").strip()
        if emotion.lower() == "none":
            emotion = ""
        # style selects a voice persona (child, older, ...); timbre comes from
        # the reference audio in IndexTTS, so style has no mapping here.
        with asset_as_path(input.audio) as spk_path:
            raw = self._synthesize(
                text=text,
                spk_audio_path=str(spk_path),
                lang=_lang(None),
                emo_text=emotion or None,
            )
        return TextAudioGenSpeechOutput(
            success=True, audio=asset(raw, mime="audio/wav")
        )

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
