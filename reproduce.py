"""Face neuron community in SDXS-512 (IDKiro/sdxs-512-dreamshaper): reproduce base | community | full edit.

What it does, for one prompt and seed:
  base       the unmodified model.
  community  base prompt and base text encoder; only 2,000 feed-forward neurons in the UNet's 16x16 layers (face_community.json)
             are set to the values they take when the people edit is on. Every other neuron and weight is base.
  edit       the full people edit: 768 values added to the text encoder's final_layer_norm.bias (people_bias_delta.npy).
Needs: torch, diffusers, transformers, numpy, pillow. Files next to this script: face_community.json, people_bias_delta.npy.
usage: python reproduce.py "a young man standing in a field at dusk holding a lantern" --seed 800 [--model IDKiro/sdxs-512-dreamshaper]
"""
import argparse, json, os, numpy as np, torch
from diffusers import UNet2DConditionModel, AutoencoderTiny
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('prompt'); ap.add_argument('--seed', type=int, default=800)
ap.add_argument('--model', default='IDKiro/sdxs-512-dreamshaper'); ap.add_argument('--out', default='.'); a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; dt = torch.float16 if dev == 'cuda' else torch.float32
tok = CLIPTokenizer.from_pretrained(a.model, subfolder='tokenizer'); te = CLIPTextModel.from_pretrained(a.model, subfolder='text_encoder')
unet = UNet2DConditionModel.from_pretrained(a.model, subfolder='unet'); vae = AutoencoderTiny.from_pretrained(a.model, subfolder='vae')
for m in (te, unet, vae): m.to(dev, dtype=dt).eval().requires_grad_(False)
COMM = json.load(open(f'{HERE}/face_community.json')); DELTA = torch.tensor(np.load(f'{HERE}/people_bias_delta.npy'), device=dev, dtype=dt)
ln = [m for n, m in te.named_modules() if n.endswith('final_layer_norm')][0]; bias0 = ln.bias.data.clone()
ab = torch.cumprod(1 - torch.linspace(.00085 ** .5, .012 ** .5, 1000) ** 2, 0)[999].item()
ids = tok(a.prompt, padding='max_length', truncation=True, max_length=77, return_tensors='pt').input_ids.to(dev)
@torch.no_grad()
def cond(edit):
    ln.bias.data.copy_(bias0 + DELTA if edit else bias0); c = te(ids)[0]; ln.bias.data.copy_(bias0); return c
mode, cache = {'v': None}, {}
def hook(name, idx):
    def f(_, inp):
        x = inp[0]
        if mode['v'] == 'capture': cache[name] = x.detach().clone()
        elif mode['v'] == 'inject': x = x.clone(); x[..., idx] = cache[name][..., idx]; return (x,)
    return f
for name, idx in COMM.items(): unet.get_submodule(name).register_forward_pre_hook(hook(name, torch.tensor(idx, device=dev)))
@torch.no_grad()
def render(c):
    z = torch.randn((1, 4, 64, 64), generator=torch.Generator('cpu').manual_seed(a.seed)).to(dev, dtype=dt)
    eps = unet(z, 999, encoder_hidden_states=c).sample; x0 = (z - (1 - ab) ** .5 * eps) / ab ** .5
    x = vae.decode(x0 / vae.config.scaling_factor).sample[0]
    return Image.fromarray(((x.float() / 2 + .5).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy())
cb, ce = cond(False), cond(True)
mode['v'] = None; base = render(cb)
mode['v'] = 'capture'; edit = render(ce)            # full edit; also records the community's edited values
mode['v'] = 'inject'; comm = render(cb)             # base prompt, community neurons set to those values
mode['v'] = None
os.makedirs(a.out, exist_ok=True); stem = f'{a.out}/seed{a.seed}'
for lab, im in (('base', base), ('community', comm), ('edit', edit)): im.save(f'{stem}_{lab}.png')
W = Image.new('RGB', (1536, 512)); [W.paste(im, (i * 512, 0)) for i, im in enumerate((base, comm, edit))]; W.save(f'{stem}_base_community_edit.png')
print(f'saved {stem}_base_community_edit.png  (neurons: {sum(len(v) for v in COMM.values())} in {len(COMM)} blocks)')
