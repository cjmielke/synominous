import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch

from train_classifier import Classifier

encoder_default = 'xcit_nano_12_p16_224.fb_in1k'
encoder_default = 'deit3_base_patch16_224'


parser = argparse.ArgumentParser()
parser.add_argument('--output_embeddings', default=None)
parser.add_argument('--color', action='store_true')
parser.add_argument('--timm', default=encoder_default, type=str)
parser.add_argument('--custom', type=str, help="weights from custom trained model")
parser.add_argument('--search')
args = parser.parse_args()

if args.search:
    for m in timm.list_models(f'*{args.search}*'):
        print(f'{m}')
    sys.exit()

if args.output_embeddings is None:
    args.output_embeddings = args.timm

output_path = args.output_embeddings
if not args.color:
    args.output_embeddings += '_grey'

output_path = Path('./data/embeddings') / args.output_embeddings
output_path.mkdir(exist_ok=True)

if args.custom is not None:
    model = Classifier(args.timm, pretrained=False)
    state_dict = torch.load(args.custom)
    if 'state_dict' in state_dict:      # is this a lightning checkpoint?
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model = model.encoder       # want embeddings, not classification head
else:
    # just using an off-the-shelf imagenet trained model for now
    model = timm.create_model(args.timm, pretrained=True, num_classes=0).cuda()

model = model.eval().cuda()


transform = transforms.Compose([
    #transforms.Resize((224, 224)), # Resize images
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


#image_dataset = datasets.ImageFolder(root=COLOR_PNG_PATH, transform=transform)

class SimplerDataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        filename = self.image_paths[index]
        img = Image.open(filename)#.convert('RGB')
        if img.mode == 'L':
            img = img.convert('RGB')
        target = filename.stem
        img = transform(img)
        return img, target

COLOR_PNG_PATH=('./tiles/0')

if args.color: img_path = Path('./data/tiles/0')
else: img_path = Path('./data/greyscale_tiles')

img_paths = list(img_path.glob('*.png'))

existing_embeddings = {int(p.stem) for p in output_path.glob('*.pt')}
total_imgs = len(img_paths)
img_paths = [i for i in img_paths if int(i.stem) not in existing_embeddings]

print(f'Need to featurize {len(img_paths)} / {total_imgs}')

image_dataset = SimplerDataset(img_paths)
dataloader = DataLoader(image_dataset, num_workers=4, batch_size=32, shuffle=False)

for batch in tqdm(dataloader):
    images, ids = batch
    embeddings = model(images.cuda()).detach().cpu()#.numpy()
    for syn_id, embedding in zip(ids, embeddings):
        #print(syn_id, embedding.shape)
        outfile = output_path / f"{syn_id}.pt"
        torch.save(embedding, outfile)

