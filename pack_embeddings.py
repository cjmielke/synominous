import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from download import get_annotated_synapses, NEUROTRANSMITTERS
import torch
import numpy as np

synapses_df = get_annotated_synapses()

parser = argparse.ArgumentParser()
parser.add_argument('embeddings_dir')
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args()

embeddings_dir = Path(args.embeddings_dir)


embeddings_available = set([int(p.stem) for p in embeddings_dir.glob('*.pt')])
synapses_df = synapses_df[synapses_df.index.isin(embeddings_available)]

if args.limit:
    synapses_df = synapses_df.sample(frac=1).head(args.limit)

synapses_df = synapses_df.reset_index()



def chunker(iterable, chunk_size):
    """Yield successive chunks from an iterable."""

    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def load_embedding_for_synapse(S: pd.Series):
    pt_file = embeddings_dir / f'{S.synapse}.pt'
    a = torch.load(pt_file).detach().cpu().numpy().astype(np.float16)
    return a

def save_json():

    #of = open('embeddings.json', 'w')
    of = gzip.open('embeddings.json.gz', 'wt')
    of.write(f'{{"embedding_name" : "{embeddings_dir.name}", "embeddings" : [')

    for idx, S in tqdm(synapses_df.iterrows(), total=len(synapses_df)):
        pt_file = embeddings_dir / f'{S.synapse}.pt'
        e = torch.load(pt_file).detach()
        v = [round(float(i), 3) for i in e.numpy()]
        #a = load_embedding_for_synapse(S)
        #v = [round(float(i), 3) for i in a]
        if idx==0:
            of.write('\n')
        else:
            of.write(',\n')
        of.write(f'  {json.dumps(dict(s=S.synapse,n=S.n,v=v))}')

    of.write('\n]\n}')
    of.close()



def split_dataframe(df, num_chunks):
  """Splits a Pandas DataFrame into a specified number of chunks.

  Args:
    df: The DataFrame to split.
    num_chunks: The number of chunks to create.

  Returns:
    A list of DataFrame chunks.
  """
  return np.array_split(df, num_chunks)



def save_parquet():
    import pyarrow as pa
    from pyarrow import Table
    import pyarrow.parquet as pq

    chunks = split_dataframe(synapses_df, 1000)
    first_chunk=True
    for chunk in tqdm(chunks):
        chunk: pd.DataFrame
        df = chunk[['synapse', 'n']].copy()
        df['v'] = df.apply(load_embedding_for_synapse, axis=1)
        #print(type(chunk))
        #print(f"Chunk {i + 1}:\n{chunk}\n")
        #print(f"Chunk {i + 1}:\n{df}\n")

        table_chunk = Table.from_pandas(df)

        if first_chunk:
            writer = pq.ParquetWriter('embeddings.parquet', table_chunk.schema)
            first_chunk = False

        writer.write_table(table_chunk)


    writer.close()


if __name__ == '__main__':
    #save_parquet()     # files were somehow bigger than json.gz!
    save_json()

