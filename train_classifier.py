import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
from torchvision import transforms
from download import get_annotated_synapses, NEUROTRANSMITTERS

import seaborn as sns
from matplotlib import pyplot as plt

mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]

default_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std)
])


train_transform = transforms.Compose([
    #transforms.Resize((224, 224)), # Resize images
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(224),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean, std=std)
])



class SynapseDataset(Dataset):

    def __init__(self, tiles_path: Path, synapse_df, transform=None):
        # filter dataframe for synapses we actually have images for
        self.tiles_path = tiles_path
        self.synapse_df = synapse_df.reset_index()
        self.transform = transform or default_transform

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, item):
        S = self.synapse_df.iloc[item]
        tile_file = self.tiles_path / f'{S.synapse}.png'
        img = Image.open(tile_file)
        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, S.n


class Classifier(pl.LightningModule):
    def __init__(self, timm_encoder, *args, pretrained=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = timm.create_model(timm_encoder, pretrained=pretrained, num_classes=0)
        #self.embedding_size = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.num_features, 6)
        )
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=6)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=6)

    def forward(self, tiles):
        embedding = self.encoder(tiles)
        probs = self.classifier(embedding)
        return probs#.detach().cpu()

    def training_step(self, batch):
        tiles, neurotransmitters = batch
        probs = self.forward(tiles)
        loss = self.loss(probs, neurotransmitters)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        tiles, neurotransmitters = batch
        probs = self.forward(tiles)
        loss = self.loss(probs, neurotransmitters)
        self.accuracy(probs, neurotransmitters)
        self.confmat.update(probs, neurotransmitters)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        confmat = self.confmat.compute()
        class_names = list(NEUROTRANSMITTERS.keys())
        num_classes = 6

        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=class_names, columns=class_names)

        print('Num of val samples: {}. Check this aligns with the numbers from the dataloader'.format(df_cm.sum(axis=1).sum() ))
        # df_cm.to_csv('raw_nums.csv') # you can use this to validate the number of samples is correct

        #normalise the confusion matrix
        norm =  np.sum(df_cm, axis=1)
        normalized_cm = (df_cm.T/norm).T #
        #validate the confusion matrix sums to num of classes
        if normalized_cm.sum(axis=1).sum() != num_classes:
          print('Error with confusion matrix')
          sys.exit()

        normalized_cm.to_csv('norm_cdf.csv') #saved locally so that I could validate outside of wandb

        f, ax = plt.subplots(figsize = (15,10))
        sns.heatmap(normalized_cm, annot=True, ax=ax)
        self.logger.experiment.log({"plot": wandb.Image(f) })

        self.confmat.reset()  #This was NEEDED otherwise the confusion matrix kept stacking the results after each epoch


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

'''
synapses_df = get_annotated_synapses()
neurotransmitter_counts = synapses_df['n'].value_counts()
weight_per_class = neurotransmitter_counts.sum() / neurotransmitter_counts
print(neurotransmitter_counts)
print(weight_per_class)
sys.exit()
'''


def train(args):

    synapses_df = get_annotated_synapses()
    # filter out synapses we haven't downloaded yet
    tiles_path = Path(args.tiles_path)
    tile_png_ids = set(int(p.stem) for p in tiles_path.glob('*.png'))
    synapses_df = synapses_df[synapses_df.index.isin(tile_png_ids)]

    neurotransmitter_counts = synapses_df['n'].value_counts()
    weight_per_class: pd.Series = neurotransmitter_counts.sum() / neurotransmitter_counts
    print(f'weight_per_class : \n{weight_per_class}')

    logger = WandbLogger(project='synominous')

    model = Classifier(args.timm).cuda()
    if args.weights:
        state_dict = torch.load(args.weights)
        if 'state_dict' in state_dict:
            print('Loading weights only from a lightning checkpoint')
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

    train_df, val_df = train_test_split(synapses_df)

    # weights for stratified sampling
    weight_per_class = weight_per_class.apply(np.log)       # dont over-weigh the really unbalanced classes
    weights = train_df['n'].apply(weight_per_class.get)
    print(weights.value_counts())
    weights = torch.from_numpy(weights.to_numpy())

    train_dataset = SynapseDataset(tiles_path, train_df, transform=train_transform)
    val_dataset = SynapseDataset(tiles_path, val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2,
                                  sampler=sampler.WeightedRandomSampler(weights, len(weights)))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    trainer = pl.Trainer(accelerator="gpu", logger=logger, max_epochs=30, val_check_interval=0.5)
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        pass

    #torch.save(model.encoder.state_dict(), "tile_encoder_weights.pth")
    torch.save(model.state_dict(), f"{args.timm}_classifier_weights.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tiles_path')
    parser.add_argument('--timm', default='deit3_base_patch16_224')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weights')
    args = parser.parse_args()

    train(args)


