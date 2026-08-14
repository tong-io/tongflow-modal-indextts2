"""Modal download entry for indextts2 (IndexTTS-2.5).

Run:
  modal run download.py::download

Self-contained: do not import other local modules.

Besides the main checkpoint repo, IndexTTS-2.5 needs four auxiliary models
that its runtime otherwise downloads at startup into ``{model_dir}/hf_cache/``
(see indextts/utils/model_download.py upstream). Pre-populating them here in
the exact same layout keeps container cold starts download-free.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import modal


_cfg: dict[str, Any] = {}


def _repo_id() -> str:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and repos:
        r = repos[0]
        if isinstance(r, dict) and r.get("repoId"):
            return str(r["repoId"])
    return "IndexTeam/IndexTTS-2.5"


volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==1.6.0"),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = _repo_id()
    model_dir = f"/models/{repo_id}"
    if not (os.path.exists(model_dir) and os.listdir(model_dir)):
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Model downloaded to {model_dir}")
    else:
        print(f"Model already exists at {model_dir}, skipping")

    cache_dir = os.path.join(model_dir, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    w2v_dir = os.path.join(cache_dir, "w2v-bert-2.0")
    if not (os.path.isdir(w2v_dir) and os.listdir(w2v_dir)):
        snapshot_download(
            repo_id="facebook/w2v-bert-2.0",
            local_dir=w2v_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"w2v-bert-2.0 downloaded to {w2v_dir}")

    for repo, remote_file, local_name in (
        ("amphion/MaskGCT", "semantic_codec/model.safetensors", "semantic_codec_model.safetensors"),
        ("funasr/campplus", "campplus_cn_common.bin", "campplus_cn_common.bin"),
    ):
        local_path = os.path.join(cache_dir, local_name)
        if os.path.isfile(local_path):
            continue
        downloaded = hf_hub_download(repo_id=repo, filename=remote_file, local_dir=cache_dir)
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            shutil.copy2(downloaded, local_path)
        print(f"{repo}/{remote_file} -> {local_path}")

    bigvgan_dir = os.path.join(cache_dir, "bigvgan")
    os.makedirs(bigvgan_dir, exist_ok=True)
    for fname in ("config.json", "bigvgan_generator.pt"):
        local_path = os.path.join(bigvgan_dir, fname)
        if os.path.isfile(local_path):
            continue
        downloaded = hf_hub_download(
            repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
            filename=fname,
            local_dir=bigvgan_dir,
        )
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            shutil.copy2(downloaded, local_path)
        print(f"bigvgan/{fname} -> {local_path}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
