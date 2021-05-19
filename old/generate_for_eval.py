import argparse
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024, DALLE
from dalle_pytorch.custom_tokenizer import tokenize, tokenizer, VOCAB_SIZE

# pandas for captions
import pandas as pd

# mem leak debug
import gc

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--dalle_path', type = str, required = True,
                    help='path to your trained DALL-E')

#parser.add_argument('--text', type = str, required = True,
#                    help='your text prompt')

parser.add_argument('--num_images', type = int, default = 128, required = False,
                    help='number of images')

parser.add_argument('--batch_size', type = int, default = 4, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--taming', dest='taming', action='store_true')

args = parser.parse_args()

# load DALL-E

dalle_path = Path(args.dalle_path)

assert dalle_path.exists(), 'trained DALL-E must exist'

load_obj = torch.load(str(dalle_path))
dalle_params, vae_params, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')

dalle_params.pop('vae', None) # cleanup later

if vae_params is not None:
    vae = DiscreteVAE(**vae_params)
elif not args.taming:
    vae = OpenAIDiscreteVAE()
else:
    vae = VQGanVAE1024()


dalle = DALLE(vae = vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

# ============
# get captions
cap_df = pd.read_pickle("./cub_2011_test_captions.pkl")
image_size = vae.image_size
# generate images
for i, row in cap_df.iterrows():
    temp_text = tokenize([row["caption"]], dalle.text_seq_len)#.cuda()  #,True,True).cuda()
    if i == 0:
        text = temp_text
        continue
    text = torch.cat((text, temp_text),0)

#text = repeat(text, '() n -> b n', b = args.num_images)
# create masks
#mask = text != 0

#dump_every_n_batches = 10


def generate_batch(bb, text):
    outputs = []
    mask = text != 0    
    #with torch.no_grad():
    for i,(text_chunk, mask) in tqdm(enumerate(zip(text.split(args.batch_size), mask.split(args.batch_size))), desc = f'generating images, {bb}'):
        output = dalle.generate_images(text_chunk, mask = mask, filter_thres = args.top_k)
        outputs.append(output)

    outputs = torch.cat(outputs)

    outputs_dir = Path(args.outputs_dir) / 'test_eval'
    outputs_dir.mkdir(parents = True, exist_ok = True)

    for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
        save_image(image, outputs_dir / f'{bb}-{i}.jpg', normalize=True)

    print(f'created {args.num_images} images at "{str(outputs_dir)}"')


big_batch = 30
max_idx = len(text)  # non-inclusive
print("len: ", max_idx)

for bb in range(1000):
    s = bb*big_batch
    e = (bb+1)*big_batch
    if e > max_idx:
        e = max_idx
    generate_batch(bb, text[s:e].cuda())

'''if i % dump_every_n_batches == 0:
            #print("Saving generated images to disk...")
            outputs = torch.cat(outputs)
            outputs_dir = Path(args.outputs_dir) / "cub-eval"
            outputs_dir.mkdir(parents = True, exist_ok = True)

            for j, image in tqdm(enumerate(outputs), desc = 'saving images'):
                save_image(image, outputs_dir / f'{i}-{j}.jpg', normalize=True)
            
            del outputs
            outputs = []

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    del obj
                else:
                    continue
            except Exception as e:
                pass
        torch.cuda.empty_cache()'''

    #print("generated so far: ", i * args.batch_size)
#print(f'created {args.num_images} images at "{str(outputs_dir)}"')

