# Foundation Model for Histopathology
==============================
Creating a foundation model for pathology using self-supervised learning.

# Datasets
1) DigestPath (both signet ring cell and colonoscopy tissue segment)
2) WSSS4LUAD

# Setup

To make it easy for you to get started with our model, here's a list of recommended next steps:

- [ ] Clone this repository into a local folder.
```

cd local/path
git clone https://gitlab.lrz.de/waly/fm_histopathology.git
```
- [ ] Setup the python virtual environement using `conda`.

```
module load python/anaconda3
cd environments 
conda env create -f ./environments/20240322.yml

```

# Model Architecture
1) ViT


# Hyperparameter search with Wandb Sweeps
Initialize Sweep project from sweep_config.yaml file 
```
wandb sweep --project sweeps_mesh sweeps_config.yaml
```

# Model Training
```
cd src/models/[model_name]
python ./train.py


```
# Results


- [ ] Check the playground notebooks for usage examples
```


```
# Authors and acknowledgment
```

```
# License
```

```
## Project status
```
