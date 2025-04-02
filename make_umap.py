import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from umap import UMAP

from download import get_annotated_synapses

default_embeddings = 'data/embeddings/custom_deit3_grey_autoencoded_32/'
default_embeddings = 'data/embeddings/custom_deit3_grey/'
default_embeddings = None

parser =  argparse.ArgumentParser()
parser.add_argument('embeddings_path')
#parser.add_argument('--embeddings_path', default=default_embeddings)
args = parser.parse_args()

synapses_df = get_annotated_synapses()

embeddings_path = Path(args.embeddings_path)
embeddings_available = set([int(p.stem) for p in embeddings_path.glob('*.pt')])

synapses_df = synapses_df[synapses_df.index.isin(embeddings_available)].sample(frac=1)

#synapses_df = synapses_df.head(100)

rows=[]
for synapse_id, row in synapses_df.iterrows():
    pt_file = embeddings_path / f'{synapse_id}.pt'
    embedding = torch.load(str(pt_file)).numpy()
    rows.append(dict(synapse=synapse_id, embedding=embedding, n=row.n, neurotransmitter=row.neurotransmitters))

df = pd.DataFrame(rows)
print(df)

embeddings = np.asarray(list(df['embedding']))
print(embeddings.shape)

umap = UMAP()
um = umap.fit_transform(embeddings)
um_x, um_y = um[:,0], um[:,1]
um_x -= um_x.mean()
um_y -= um_y.mean()


umap_df = pd.DataFrame(dict(s=df.synapse, n=df['n'], neurotransmitter=df['neurotransmitter'],
                            x=um_x,y=um_y))
print(umap_df)

with open('umap.json', 'w') as fh:
    fh.write(umap_df.to_json(orient='records'))

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(rc={'figure.figsize':(12,12)})
ax = sns.scatterplot(data=umap_df, x='x', y='y', hue='neurotransmitter', s=5)
ax.legend(markerscale=3)
#for lh in ax.legend.legendHandles:
#    lh.set_alpha(1)
#    lh._sizes = [10]
ax.set_title(f'UMAP_{embeddings_path.name}')
plt.savefig(f'scatterplot.png')
plt.savefig(f'img/umaps/UMAP_{embeddings_path.name}.png')


