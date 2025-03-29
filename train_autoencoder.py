import argparse
from glob import iglob
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, noise=0):
        self.embedding_pt_files = list(Path(embeddings_dir).glob('*.pt'))
        self.noise = noise
        emb0, syn0 = self[0]
        self.embedding_dim = len(emb0)

    def __len__(self):
        return len(self.embedding_pt_files)

    def __getitem__(self, item):
        pt_file = self.embedding_pt_files[item]
        tensor = torch.load(str(pt_file))
        if self.noise:
            tensor = tensor + self.noise * torch.randn(tensor.shape)
        synapse_id = pt_file.stem
        return tensor, synapse_id



class AutoEncoder(LightningModule):

    def __init__(self, embSize=1024, nHidden=128, wandb=None):
        super().__init__()

        self.wandb = wandb
        self.encoder = nn.Linear(embSize, nHidden)
        self.decoder = nn.Linear(nHidden, embSize)
        self.criterion = MSELoss()

    def getFeatures(self, x):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        xH = self.decoder(h)
        return xH

    def step(self, batch, batch_index):
        latents, synapse_ids = batch
        l = self.forward(latents)
        loss = self.criterion(l, latents)
        return dict(loss=loss)

    def training_step(self, batch, batch_index):
        out = self.step(batch, batch_index)
        self.log('train_loss', out['loss'])
        return out

    def validation_step(self, batch, batch_index):
        out = self.step(batch, batch_index)
        self.log('val_loss', out['loss'])
        return out

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.01)
        return optim



def train(args, dataset, autoencoder):

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator='gpu',
                         log_every_n_steps=10,
                         #accumulate_grad_batches=64
                        )

    try:
        trainer.fit(autoencoder, dataloader)
    except KeyboardInterrupt:
        pass        # save weights on ctrl-c

    #torch.save(autoencoder.encoder.state_dict(), "encoder_weights.pt")
    torch.save(autoencoder.state_dict(), "autoencoder_weights.pt")

def infer(args, dataset, autoencoder):
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for tensor, synapse_id in tqdm(dataset):
        h = autoencoder.encoder(tensor).detach().cpu()
        outfile = str(out_path / f"{synapse_id}.pt")
        torch.save(h, outfile)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'infer'])
    parser.add_argument('embeddings_dir')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--out_path', type=str, default='./autoencoder_embeddings')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    dataset = EmbeddingDataset(args.embeddings_dir)
    autoencoder = AutoEncoder(dataset.embedding_dim, nHidden=32)

    if args.weights:
        autoencoder.load_state_dict(torch.load(args.weights))

    if args.command == 'train':
        train(args, dataset, autoencoder)
    elif args.command == 'infer':
        assert args.weights is not None
        infer(args, dataset, autoencoder)
