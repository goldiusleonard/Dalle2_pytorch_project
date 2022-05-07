import torch
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms as T
from pathlib import Path
import os
from tqdm import tqdm
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter
from dalle2_pytorch.tokenizer import SimpleTokenizer
from dalle2_pytorch.optimizer import get_optimizer
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
train_annot_path = "./coco_annotations/captions_train2014.json"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your diffusion prior model save path here (end with ".pth")
diff_save_path = "./diff_prior.pth"

# Change your diffusion prior model save path here (end with ".pth")
decoder_save_path = "./decoder.pth"

# Change the model weight save path here (end with ".pth")
dalle2_save_path = "./dalle2.pth"

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

train_data = CocoCaptions(
    root=train_img_path,
    annFile=train_annot_path,
    transform=transform
)

# openai pretrained clip - defaults to ViT/B-32
OpenAIClip = OpenAIClipAdapter()

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
)

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = OpenAIClip,
    timesteps = 100,
    cond_drop_prob = 0.2
).to(device)

unet = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8)
).to(device)

# decoder, which contains the unet and clip

decoder = Decoder(
    unet = unet,
    clip = OpenAIClip,
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5,
    condition_on_text_encodings=True
).to(device)

if os.path.exists(diff_save_path):
    diffusion_prior.load_state_dict(torch.load(diff_save_path))

if os.path.exists(decoder_save_path):
    decoder.load_state_dict(torch.load(decoder_save_path))

train_size = len(train_data)
idx_list = range(0, train_size, batch_size)

tokenizer = SimpleTokenizer()
opt = get_optimizer(diffusion_prior.parameters())
sched = ExponentialLR(opt, gamma=0.01)

for curr_epoch in range(epoch):
    print("Run training diffusion prior ...")
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
            image = image.unsqueeze(0).to(device)

            texts = tokenizer.tokenize(target).to(device)

            for text in texts:
                if total_loss == torch.tensor(0., device=device):
                    total_loss = diffusion_prior(text.unsqueeze(0), image)
                else:
                    total_loss += diffusion_prior(text.unsqueeze(0), image)
                batch_len += 1
                
        avg_loss = total_loss / batch_len

        opt.zero_grad()
        avg_loss.backward()
        opt.step()

        if batch_idx % 100 == 0:
            torch.save(diffusion_prior.state_dict(), diff_save_path)
            print(f"average loss: {avg_loss.data}")
        
    sched.step()

torch.save(diffusion_prior.state_dict(), diff_save_path)

opt = get_optimizer(decoder.parameters())
sched = ExponentialLR(opt, gamma=0.01)

for curr_epoch in range(epoch):
    print("Run training decoder ...")
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
            image = image.unsqueeze(0).to(device)

            texts = tokenizer.tokenize(target).to(device)

            for text in texts:
                if total_loss == torch.tensor(0., device=device):
                    total_loss = decoder(text.unsqueeze(0), image)
                else:
                    total_loss += decoder(text.unsqueeze(0), image)
                batch_len += 1
                
        avg_loss = total_loss / batch_len

        opt.zero_grad()
        avg_loss.backward()
        opt.step()

        if batch_idx % 100 == 0:
            torch.save(decoder.state_dict(), decoder_save_path)
            print(f"average loss: {avg_loss.data}")
        
    sched.step()

torch.save(decoder.state_dict(), decoder_save_path)

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
).to(device)

torch.save(dalle2.state_dict(), dalle2_save_path)