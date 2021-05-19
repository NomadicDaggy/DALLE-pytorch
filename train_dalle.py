import argparse
from random import choice
from pathlib import Path
import numpy as np

# torch

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import distributed_utils
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer

# argument parsing

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required = False)

group.add_argument('--vae_path', type = str,
                    help='path to your trained discrete VAE')

group.add_argument('--dalle_path', type = str,
                    help='path to your partially trained DALL-E')

parser.add_argument('--image_text_folder', type = str, required = True,
                    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument('--truncate_captions', dest='truncate_captions',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type = float, default = 0.6,
                    help='Random resized crop lower ratio')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--fp16', action='store_true')

parser.add_argument('--learning_rate')

parser = distributed_utils.wrap_arg_parser(parser)

args = parser.parse_args()

# helpers

def exists(val):
    return val is not None

# constants

VAE_PATH = args.vae_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = 5
BATCH_SIZE = 16
#LEARNING_RATE = 4.5e-4
LEARNING_RATE = float(args.learning_rate)

GRAD_CLIP_NORM = 0

MODEL_DIM = 256
TEXT_SEQ_LEN = 80
DEPTH = 8
HEADS = 8
DIM_HEAD = 64

REVERSIBLE = False
LOSS_IMG_WEIGHT = 7

ATTN_TYPES = ['full','axial_row','axial_col','conv_like']
#ATTN_TYPES = ['axial_row', 'axial_col']

LR_DECAY = True
LR_DECAY_FACTOR = 0.5
LR_DECAY_PATIENCE = 5
LR_DECAY_COOLDOWN = 0
LR_DECAY_MIN = 1e-7

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# tokenizer

if exists(args.bpe_path):
    tokenizer = HugTokenizer(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# reconstitute vae

if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    else:
        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()
        
    dalle_params = dict(        
        **dalle_params
    )
    IMAGE_SIZE = vae.image_size
else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        if distr_backend.is_root_worker():
            print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
        vae = vae_klass()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens = tokenizer.vocab_size,
        text_seq_len = TEXT_SEQ_LEN,
        dim = MODEL_DIM,
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD,
        reversible = REVERSIBLE,
        loss_img_weight = LOSS_IMG_WEIGHT,
        attn_types = ATTN_TYPES,
    )

# configure OpenAI VAE for float16s

if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    vae.enc.blocks.output.conv.use_float16 = True

# helpers

def save_model(path):
    if not distr_backend.is_root_worker():
        return

    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'weights': dalle.state_dict()
    }

    torch.save(save_obj, path)

def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

# dataset loading

class TextImageDataset(Dataset):
    def __init__(self, folder, text_len = 256, image_size = 128):
        super().__init__()
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]

        image_files = [
            *path.glob('**/*.png'),
            *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'),
            *path.glob('**/*.bmp')
        ]

        text_files = {t.stem: t for t in text_files}
        image_files = {i.stem: i for i in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len

        self.image_tranform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale = (args.resize_ratio, 1.), ratio = (1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        text_file = self.text_files[key]
        image_file = self.image_files[key]

        image = Image.open(image_file)
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = choice(descriptions)

        tokenized_text = tokenizer.tokenize(description, self.text_len, truncate_text=args.truncate_captions).squeeze(0)

        image_tensor = self.image_tranform(image)
        return tokenized_text, image_tensor

# create dataset and dataloader

ds = TextImageDataset(
    args.image_text_folder,
    text_len = TEXT_SEQ_LEN,
    image_size = IMAGE_SIZE
)

assert len(ds) > 0, 'dataset is empty'
if distr_backend.is_root_worker():
    print(f'{len(ds)} image-text pairs found for training')

if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank())
else:
    data_sampler = None

dl = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = not data_sampler,
                drop_last = True, sampler=data_sampler)

# initialize DALL-E

dalle = DALLE(vae = vae, **dalle_params)
if args.fp16:
    dalle = dalle.half()
dalle = dalle.cuda()


if RESUME:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)

if LR_DECAY:
    scheduler = ReduceLROnPlateau(
        opt,
        mode = "min",
        factor = LR_DECAY_FACTOR,
        patience = LR_DECAY_PATIENCE,
        cooldown = LR_DECAY_COOLDOWN,
        min_lr = LR_DECAY_MIN,
        verbose = True,
    )

if distr_backend.is_root_worker():
    # experiment tracker

    import wandb

    model_config = dict(
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        grad_clip_norm = GRAD_CLIP_NORM,
        model_dim = MODEL_DIM,
        text_seq_len = TEXT_SEQ_LEN,
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD,
        reversible = REVERSIBLE,
        loss_img_weight = LOSS_IMG_WEIGHT,
        lr_decay_factor = LR_DECAY_FACTOR,
        lr_decay_patience = LR_DECAY_PATIENCE,
        lr_decay_cooldown = LR_DECAY_COOLDOWN,
        attn_types = ATTN_TYPES,
        resize_ratio = args.resize_ratio,
    )

    run = wandb.init(
        project = 'dalle_train_CUB_proper',
        resume = RESUME,
        config = model_config,
        save_code=True,
        #id="worldly_salad_no_10epoch",
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {
    'train_batch_size': BATCH_SIZE,
    'gradient_clipping': GRAD_CLIP_NORM,
    'fp16': {
        'enabled': args.fp16,
    },
}

(distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=dalle.parameters(),
    training_data=ds if using_deepspeed else dl,
    lr_scheduler=scheduler if LR_DECAY else None,
    config_params=deepspeed_config,
)
avoid_model_calls = using_deepspeed and args.fp16

LOG_FILENAME = f"{run.name}.txt"

f = open(LOG_FILENAME, "a+")

# training
torch.cuda.empty_cache() # Avoid allocation error due to potential bug in deepspeed. See https://github.com/lucidrains/DALLE-pytorch/issues/161
for epoch in range(EPOCHS):
    for i, (text, images) in enumerate(distr_dl):
        if args.fp16:
            images = images.half()
        text, images = map(lambda t: t.cuda(), (text, images))

        loss = distr_dalle(text, images, return_loss = True)

        if using_deepspeed:
            distr_dalle.backward(loss)
            distr_dalle.step()
            # Gradients are automatically zeroed after the step
        else:
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
            distr_opt.step()
            distr_opt.zero_grad()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)
        f.write(f"{epoch} {i} {avg_loss.item()} {opt.param_groups[0]['lr']}\n")
        if distr_backend.is_root_worker():
            log = {}

            if i % 10 == 0:
                print(epoch, i, f'loss - {avg_loss.item()}')

                log = {
                    **log,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item(),
                    'lr': opt.param_groups[0]['lr'],
                }

                f.close()
                f = open(LOG_FILENAME, "a+")

            if i % 100 == 0:
                sample_text = text[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)

                if not avoid_model_calls:
                    # CUDA index errors when we don't guard this
                    image = dalle.generate_images(text[:1], filter_thres = 0.9) # topk sampling at 0.9

                save_model(f'./dalle.pt')
                #wandb.save(f'./dalle.pt')

                log = {
                    **log,
                }
                if not avoid_model_calls:
                    log['image'] = wandb.Image(image, caption = decoded_text)
            wandb.log(log)

    if LR_DECAY:
        distr_scheduler.step(loss)

    #if distr_backend.is_root_worker():
    #    # save trained model to wandb as an artifact every epoch's end
    #
    #    model_artifact = wandb.Artifact('trained-dalle', type = 'model', metadata = dict(model_config))
    #    model_artifact.add_file('dalle.pt')
    #    run.log_artifact(model_artifact)

    if epoch %19 == 0:
        save_model(f'./sweep1/{run.name}-{epoch}.pt')

f.close()

if distr_backend.is_root_worker():
    save_model(f'./dalle-final.pt')
    wandb.save(f'./{run.name}-dalle-final.pt')
    #model_artifact = wandb.Artifact('trained-dalle', type = 'model', metadata = dict(model_config))
    #model_artifact.add_file(f'./{run.name}-dalle-final.pt')
    #run.log_artifact(model_artifact)

    wandb.finish()
