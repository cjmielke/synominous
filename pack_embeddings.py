import argparse
import json
from pathlib import Path

from tqdm import tqdm

from download import get_annotated_synapses, NEUROTRANSMITTERS
import torch

synapses_df = get_annotated_synapses()

parser = argparse.ArgumentParser()
parser.add_argument('embeddings_dir')
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args()

embeddings_dir = Path(args.embeddings_dir)

of = open('embeddings.json', 'w')
of.write('{"embeddings" : [')

embeddings_available = set([int(p.stem) for p in embeddings_dir.glob('*.pt')])
synapses_df = synapses_df[synapses_df.index.isin(embeddings_available)]



#synapses_df['n'] = synapses_df.neurotransmitters.map(NEUROTRANSMITTERS)


#df = synapses_df.sample(frac=0.1).reset_index()

if args.limit:
    synapses_df = synapses_df.sample(frac=1).head(args.limit)

df = synapses_df.reset_index()


for idx, S in tqdm(df.iterrows(), total=len(df)):
    pt_file = embeddings_dir / f'{S.synapse}.pt'
    e = torch.load(pt_file).detach()
    v = [round(float(i), 3) for i in e.numpy()]
    if idx==0:
        of.write('\n')
    else:
        of.write(',\n')
    of.write(f'  {json.dumps(dict(s=S.synapse,n=S.n,v=v))}')

of.write('\n]\n}')
of.close()

'''
for pt in Path('embeddings/').glob('*.pt'):
    synapse_id = int(pt.stem)
    if synapse_id in synapses_df:
        synapse = synapses_df.loc[synapse_id]
        e = torch.load(str(pt))
        v = [round(float(i),3) for i in e.numpy()]
        of.write(f'{json.dumps(dict(v=v))}\n')
    else:
        print('skip')
'''