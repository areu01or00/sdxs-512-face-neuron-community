# Model neuron communities

Neuron communities found by an agent (Claude Opus) inside generative models: small groups of neurons that do one job together,
and that change the output when they are switched on. Each folder is one model and one community, with the neuron list and a script to reproduce it.

| folder | model | community | what it does |
|---|---|---|---|
| [sdxs-512-face](sdxs-512-face) | [IDKiro/sdxs-512-dreamshaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper) (one-step image model) | 2,000 neurons, 8.7% of the UNet's feed-forward neurons | paints the person: turned toward you, with a clear face |

Related: [qwen-tts-neuron-mapping](https://github.com/areu01or00/qwen-tts-neuron-mapping) (human-speech neurons in Qwen3-TTS).

---

# SDXS-512: the face community

https://github.com/areu01or00/model-neuron-communities/raw/main/sdxs-512-face/demo.mp4

Left: base. Middle: base with only the face community switched on. Right: the full edit it comes from. Same prompt, same seed.
Seeds 800-803, not used to find the community, not picked.

## The neurons

2,000 GEGLU neurons in the three 16×16 transformer blocks of the UNet, where the picture's layout is decided
([face_community.json](sdxs-512-face/face_community.json), block → neuron indices):

| block | neurons | of |
|---|---|---|
| `up_blocks.0.attentions.1.transformer_blocks.0.ff.net.2` | 1,383 | 5,120 |
| `up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2` | 525 | 5,120 |
| `down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2` | 92 | 5,120 |

A neuron's activation is the input to that `ff.net.2` (the GEGLU output); index = column of `ff.net.2.weight`.
The UNet has 23,040 such neurons in all; the community is 8.7% of them.

## How it is switched on

The community comes from a people edit: 768 values added to the text encoder's `final_layer_norm.bias`
([people_bias_delta.npy](sdxs-512-face/people_bias_delta.npy)). For a prompt and seed:

1. run the model with the edit and record the community's activations;
2. run the base model (base text encoder, base prompt) and replace only those 2,000 activations with the recorded ones.

Every other neuron and every weight stays base. This is an activation-level switch; it is not yet a standalone weight patch.

## Results

Face detail = fine-scale detail inside eye-aligned face crops, in units of the base model's own seed-to-seed spread.

| | face detail | prompts with clearer faces |
|---|---|---|
| full edit | +3.25 | 12 / 12 |
| **face community alone** | **+3.05 (94%)** | **12 / 12** |

(12 prompts, 8 new seeds each, 800-807.)

It is the group, not a few strong neurons (12 held-out prompts × 4 seeds):

| switched on in base | face detail |
|---|---|
| the community (2,000) | +1.33 |
| 2,000 random neurons from the same blocks | +0.20 |
| top 50 / top 200 of the community | +0.13 / +0.37 |
| every feed-forward neuron of the UNet | +1.73 |

| switched off in the edit | face detail left |
|---|---|
| none (full edit) | +1.78 |
| the community | +0.67 |
| 2,000 random neurons from the same blocks | +1.66 |

The 32×32 blocks that follow, where faces get their fine detail, carry none of it (≈0 when switched on alone).

## Reproduce

```
pip install torch diffusers transformers numpy pillow
cd sdxs-512-face
python reproduce.py "a young man standing in a field at dusk holding a lantern" --seed 800
```

Saves `seed800_base.png`, `seed800_community.png`, `seed800_edit.png` and a side-by-side. Downloads the model from Hugging Face on first run
(`--model <path>` for a local copy). Our check: from a clean folder it matches our renders to within about 1/255 per pixel.

Prompts where it shows most: one person, small or dim in the frame, named (a man, an old woman, a boy), at dusk, in mist or under one light.
Close-up portraits already have large faces and change little.

## Limits

- One model, one community. Activation-level, not yet a weight patch.
- The community also pulls the gaze to the camera: "looking out of a window" or "reading" can end up looking at you.
- Community faces are slightly softer and paler than the full edit's.
- Asked for "a tiny figure in the distance", it still makes the figure big.
