# Anomaly Detector #1

This is a testbed for anomaly detection experiments.

## Visualize Synthetic Data 

```sh
python -m ad.data_vis
```

## Run the TensorFlow Model

A simple example might be:

```sh
python -m ad.model --model_id=1 ...
```

To specify different GPUs (for example, in a 4 GPU system):

```sh
CUDA_VISIBLE_DEVICES=0 python -m ad.model --model_id=1 ...
CUDA_VISIBLE_DEVICES=1 python -m ad.model --model_id=1 ...
CUDA_VISIBLE_DEVICES=2 python -m ad.model --model_id=1 ...
CUDA_VISIBLE_DEVICES=3 python -m ad.model --model_id=1 ...
```

## Interactive Development with Code Reloading 

```sh
conda activate ad1
python
```

In the Python interpreter, run:

```python
from ad.interact import reload
import numpy as np
np.set_printoptions(precision=4)
from ad import conn, gen, model, vis 
```

To reload code:

```python
reload()
```

## Setup

Run these commands on initial setup only:

```sh
conda create -n ad1 python=3.6
conda activate ad1
conda install numpy pandas scipy
pip install argparse matplotlib tensorflow
```

For a machine with one or more GPUs:

```
pip install tensorflow-gpu
```

To list installed packages, use:

```sh
conda list
```

If you want to remove the `ad1` environment and start from scratch:

```sh
conda deactivate
conda remove --all -n ad1
```

## Testing

Give [pytest](https://docs.pytest.org/en/latest/) a try:

```
pytest
```

Note: In a recent install with `conda install pytest`, there were errors that
may have been specific to macOS.
