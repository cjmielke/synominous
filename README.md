# SynOminous

This project is inspired by the publication ["Neurotransmitter Classification from Electron Microscopy Images at Synaptic Sites in Drosophila"](https://www.biorxiv.org/content/10.1101/2020.06.12.148775v2)

Utilizing the same dataset, presented here is a simple web-app that performs in-browser similarity searches among ~100k 2D images of synapses from a fruitfly brain. It is designed to allow the user to explore similar-looking synapses from other neuronal subtypes.

[Live demo is here](https://cjmielke.github.io/synominous/vectorsearch.html)

### Classification

Although the live demo uses an off-the-shelf imagenet vision transformer (XCIT), an attempt was made to train a classifier. I obtained similar performance to the paper, however I found that it did not seemingly perform as well for simularity searches. I hypothesize this is because the neuronal subtypes become isolated in the embedding space.

This was a quick weekend project, so I decided to train a 2D classifier despite a 3D convnet. I did however exploit the 3 RGB channels of pretrained networks to look at a stack of 3 depthwise tiles for each synapse. The images are all from the [Full Adult Fly Brain "FAFB"](https://flyconnecto.me/2023/10/18/we-mapped-the-full-adult-fly-brain/). Since the volumes don't appear to be isotropic in the Z axis, making use of existing RGB channels of pretrained networks might find some suitable depthwise patterns.

[!confusion matrix](./img/cm.png)

[!UMAP](./img/umaps/UMAP_fiery_spaceship.png)

