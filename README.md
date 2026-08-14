# tongflow-modal-indextts2

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Emotionally expressive text-to-speech with **IndexTTS-2.5** (`IndexTeam/IndexTTS-2.5`, Bilibili), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Speech synthesis — voice clone** (`text-gen-speech-clone`) — zero-shot voice cloning from a reference audio clip (Chinese, English, Japanese, Spanish, Arabic).
- **Emotive speech** (`text-audio-gen-speech`) — synthesize text in the voice of a reference audio with an emotion instruction (happy, sad, angry, …), driven by IndexTTS-2.5's emotion-text guidance.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The IndexTTS-2.5 weights and auxiliary models (w2v-bert-2.0, MaskGCT semantic codec, CAMPPlus, BigVGAN) are public — no Hugging Face token required.

## License note

IndexTTS-2.5 weights are distributed under the [bilibili IndexTTS Model License](https://github.com/index-tts/index-tts/blob/main/INDEX_MODEL_LICENSE); review it before commercial use.
