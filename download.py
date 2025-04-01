
#import gevent.monkey
#gevent.monkey.patch_all(threads=False)

import json
import sys
from random import shuffle

import pandas as pd
from tqdm import tqdm
import numpy as np
from cloudvolume import CloudVolume
from PIL import Image
import concurrent.futures
from pathlib import Path

DATA_ROOT = Path('./synister/data/fafb_v3/')

CLOUDVOLUME_URL = 'precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig'
vol = CloudVolume(CLOUDVOLUME_URL, use_https=True, green_threads=False)
vol.progress=False


NEUROTRANSMITTERS = dict(
    acetylcholine=0,
    gaba=1,
    dopamine=2,
    glutamate=3,
    serotonin=4,
    octopamine=5
)


def get_synapse_tile_at_coord(x, y, z, m = 128):
    # scale by FAFB resolution
    x, y, z = x//4, y//4, z//40

    vc=vol[x-m:x+m, y-m:y+m, z-1:z+2]
    a=np.asarray(vc).squeeze()
    return a

def download_synapse(S: pd.Series):
    #print(S)
    if type(S)==tuple: S=S[1]
    ID = S['synapse']
    #img_file = f'synapse_tiles/{ID}.png'
    img_file = f'data/tiles/0/{ID}.png'
    if Path(img_file).exists():
        #print('skip')
        return
    arr = get_synapse_tile_at_coord(S['x'], S['y'], S['z'])
    img = Image.fromarray(arr)
    img.save(img_file)



def get_annotated_synapses():

    skeletons = json.load(open(str(DATA_ROOT / 'skeletons_v3.json'), 'r'))
    rows = []
    # a single skeleton is apparently assigned gaba & octopamine .... I feel like skipping that for now
    for k,v in skeletons.items():
        if (nt:=v['nt_known']) is not None and len(nt)==1:
            rows.append(dict(skeleton=int(k), neurotransmitters=",".join(nt)))
    skeletons_df = pd.DataFrame(rows).set_index('skeleton')

    print(f'========= Neurotransmitter breakdown among {len(skeletons_df)} skeletons ==============')
    print(skeletons_df.neurotransmitters.value_counts())

    synapses = json.load(open(str(DATA_ROOT / 'synapses_v3.json'), 'r'))
    print(f'Total synapses : {len(synapses)}')

    # skeletons_df now only contains those with known neurotransmitters, so lets filter the synapses
    rows = []
    for k,v in synapses.items():
        if v['skeleton_id'] in skeletons_df.index:
            rows.append(dict(synapse=int(k), x=v['x'], y=v['y'], z=v['z'], skeleton=v['skeleton_id']))

    synapses_df = pd.DataFrame(rows).set_index('synapse')

    synapses_df = synapses_df.merge(skeletons_df, left_on='skeleton', right_index=True)

    print(f'========= Neurotransmitter breakdown among {len(synapses_df)} synapses ==============')
    print(synapses_df.neurotransmitters.value_counts())

    synapses_df['n'] = synapses_df['neurotransmitters'].map(NEUROTRANSMITTERS)

    return synapses_df


def download():

    synapses_df = get_annotated_synapses()
    # FIXME - temporarily focus on the rare ones
    synapses_df = synapses_df[synapses_df.neurotransmitters.isin(['gaba', 'dopamine','glutamate','serotonin', 'octopamine'])]
    print(synapses_df.shape)
    #sys.exit()

    # for downloading from the list of filtered synapses with assigned neurotransmitters
    already_downloaded = set([int(p.stem) for p in Path('data/tiles/0/').glob('*.png')])
    print(f'already downloaded : {len(already_downloaded)}')
    to_download = synapses_df[~synapses_df.index.isin(already_downloaded)]
    print(f'to download: {len(to_download)} / {len(synapses_df)}')

    queue = list(to_download.reset_index().iterrows())
    shuffle(queue)
    for _, row in tqdm(queue):
        download_synapse(row)


    #'''
    def run(f, my_iter):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
        return results

    #run(download_synapse, queue)
    #'''


if __name__ == '__main__':
    download()