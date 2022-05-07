import torch
from torchvision import transforms as T
from pathlib import Path
import os
from tqdm import tqdm
from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer
from torchvision.datasets.coco import CocoCaptions

# Change your input size here
input_image_size = 256

# Change your train image root path here
test_img_path = "./val2014/"

# Change your train annot json path here
test_annot_path = "./annotations/captions_val2014.json"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your dalle model path here
dalle_load_path = "./dalle.pth"

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

test_data = CocoCaptions(
    root=test_img_path,
    annFile=test_annot_path,
    transform=transform
)

tokenizer = SimpleTokenizer()

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 1024,      # codebook dimension
    hidden_dim = 64,          # hidden dimension
    num_resnet_blocks = 1,    # number of resnet blocks
    temperature = 0.9         # gumbel softmax temperature, the lower this is, the harder the discretization
).to(device)

dalle = DALLE(
    dim = 1024,
    vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = 49408,    # vocab size for text
    text_seq_len = 256,         # text sequence length
    depth = 1,                  # should aim to be 64
    heads = 16,                 # attention heads
    dim_head = 64,              # attention head dimension
    attn_dropout = 0.1,         # attention dropout
    ff_dropout = 0.1            # feedforward dropout
).to(device)

dalle.load_state_dict(torch.load(dalle_load_path))

for data in tqdm(test_data):
    _, target = data

    text = tokenizer.tokenize(target).to(device)

    test_img_tensors = dalle.generate_images(text)

    for test_idx, test_img_tensor in enumerate(test_img_tensors):
        test_img = T.ToPILImage()(test_img_tensor)
        test_save_path = test_img_save_path + "/" + str(target[test_idx]) + ".jpg"
        test_img.save(Path(test_save_path))