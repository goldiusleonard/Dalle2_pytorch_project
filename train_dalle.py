import torch
from torchvision import transforms as T
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer
from torchvision.datasets.coco import CocoCaptions

# Change your input size here
input_image_size = 256

# Change your batch size here
batch_size = 1

# Change your epoch here
epoch = 5

# Change your train image root path here
train_img_path = "./train2014/"

# Change your train annot json path here
train_annot_path = "./annotations/captions_train2014.json"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your VAE save path here (ends with ".pth")
vae_save_path = "./vae.pth"

# Change your dalle model save path here (ends with ".pth")
dalle_save_path = "./dalle.pth"

transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize(input_image_size),
    T.CenterCrop(input_image_size),
    T.ToTensor()
])

train_data = CocoCaptions(
    root=train_img_path,
    annFile=train_annot_path,
    transform=transform
)

vae = DiscreteVAE(
    image_size = 256,
    num_layers = 3,
    num_tokens = 8192,
    codebook_dim = 1024,
    hidden_dim = 64,
    num_resnet_blocks = 1,
    temperature = 0.9
).to(device)

if os.path.exists(vae_save_path):
    vae.load_state_dict(torch.load(vae_save_path))

train_size = len(train_data)
idx_list = range(0, train_size, batch_size)

tokenizer = SimpleTokenizer()
opt = Adam(
    vae.parameters(),
    lr = 3e-4,
    weight_decay=0.01,
    betas = (0.9, 0.999)
)
sched = ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=10,
    cooldown=10,
    min_lr=1e-6,
    verbose=True,
)

for curr_epoch in range(epoch):
    print("Run training discrete vae ...")
    print(f"Epoch {curr_epoch+1} / {epoch}")
    
    for batch_idx in tqdm(idx_list):
        if (batch_idx + batch_size) > train_size - 1:
            iter_idx = range(batch_idx, train_size, 1)
        else:
            iter_idx = range(batch_idx, batch_idx+batch_size, 1)

        batch_len = 0
        total_loss = torch.tensor(0., device=device)

        for curr_idx in iter_idx:
            image, _ = train_data[curr_idx]
            image = image.unsqueeze(0).type(torch.FloatTensor).to(device)

            if total_loss == torch.tensor(0., device=device):
                total_loss = vae(image, return_loss=True)
            else:
                total_loss += vae(image, return_loss=True)
            batch_len += 1
                
        avg_loss = total_loss / batch_len

        opt.zero_grad()
        avg_loss.backward()
        opt.step()

        if batch_idx % 100 == 0:
            torch.save(vae.state_dict(), vae_save_path)
            print(f"loss: {avg_loss.data}")
        
    sched.step(avg_loss)

torch.save(vae.state_dict(), vae_save_path)

tokenizer = SimpleTokenizer()

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

if os.path.exists(dalle_save_path):
    dalle.load_state_dict(torch.load(dalle_save_path))

train_size = len(train_data)
idx_list = range(0, train_size, batch_size)

opt = Adam(
    dalle.parameters(),
    lr = 3e-4,
    weight_decay=0.01,
    betas = (0.9, 0.999)
)
sched = ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=10,
    cooldown=10,
    min_lr=1e-6,
    verbose=True,
)

for curr_epoch in range(epoch):
    print("Run training dalle ...")
    print(f"Epoch {curr_epoch+1} / {epoch}")
    
    for batch_idx in tqdm(idx_list):
        if (batch_idx + batch_size) > train_size - 1:
            iter_idx = range(batch_idx, train_size, 1)
        else:
            iter_idx = range(batch_idx, batch_idx+batch_size, 1)

        batch_len = 0
        total_loss = torch.tensor(0., device=device)

        for curr_idx in iter_idx:
            image, target = train_data[curr_idx]
            image = image.unsqueeze(0).type(torch.FloatTensor).to(device)
            texts = tokenizer.tokenize(target).type(torch.LongTensor).to(device)

            for text in texts:
                if total_loss == torch.tensor(0., device=device):
                    total_loss = dalle(text.unsqueeze(0), image, return_loss=True)
                else:
                    total_loss += dalle(text.unsqueeze(0), image, return_loss=True)
                batch_len += 1
                
        avg_loss = total_loss / batch_len

        opt.zero_grad()
        avg_loss.backward()
        opt.step()

        if batch_idx % 100 == 0:
            torch.save(dalle.state_dict(), dalle_save_path)
            print(f"average loss: {avg_loss.data}")
        
    sched.step(avg_loss)

torch.save(dalle.state_dict(), dalle_save_path)