import torch
from torchvision import transforms as T
from pathlib import Path
import pandas as  pd
import os
from tqdm import tqdm
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter

# Change your input size here
input_image_size = 256

# Change your test annot csv path here
test_annot_path = "./Flower_Dataset_Combine/New_captions.csv"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change the model weight load path here (end with ".pth")
dalle2_load_path = "./dalle2.pth"

# Change the test result image save path (should be a directory or folder)
test_img_save_path = "./result"

if not os.path.exists(test_img_save_path):
    os.makedirs(test_img_save_path)

transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize(input_image_size),
    T.CenterCrop(input_image_size),
    T.ToTensor()
])

test_csv= pd.read_csv(test_annot_path)

test_csv = test_csv.drop_duplicates()
test_csv = test_csv.dropna()

# openai pretrained clip - defaults to ViT/B-32
OpenAIClip = OpenAIClipAdapter()

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).to(device)

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = OpenAIClip,
    timesteps = 100,
    cond_drop_prob = 0.2
).to(device)

unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).to(device)

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).to(device)

# decoder, which contains the unet and clip

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = OpenAIClip,
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5,
    condition_on_text_encodings=False
).to(device)

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
).to(device)

dalle2.load_state_dict(torch.load(dalle2_load_path))

for data in tqdm(test_csv.iterrows()):
    target = [data[1]['caption']]

    test_img_tensors = dalle2(
        target,
        cond_scale = 2., # classifier free guidance strength (> 1 would strengthen the condition)
    )

    for test_idx, test_img_tensor in enumerate(test_img_tensors):
        test_img = T.ToPILImage()(test_img_tensor)
        test_save_path = test_img_save_path + "/" + str(target[test_idx]) + ".jpg"
        test_img.save(Path(test_save_path))