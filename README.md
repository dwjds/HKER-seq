# HKER-Seq

Training code for the HKER-Seq hyperbolic sequential recommender on MOOCCube and MOOPer.



## Requirements

- Python 3.10
- PyTorch 2.7.0 with the CUDA 11.8 build (`2.7.0+cu118`)
- CUDA-capable GPU for full training


Create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate hker-seq
```

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

Check GPU access after installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```


## Data

The repository contains the processed interaction files used by the training code:

- `data/raw/mooccube.gz`
- `data/raw/mooper.gz`


## Training and evaluation

Run the unified launcher from the repository root:

```bash
bash scripts/train.sh mooccube
bash scripts/train.sh mooper
```

The launcher uses GPU 0 and a physical batch quota of 128 by default. These values can be overridden without changing the effective training batch size:

```bash
GPU_ID=1 BATCH_QUOTA=64 bash scripts/train.sh mooccube
```

Results and checkpoints are written to `outputs/`.



## License

See `LICENSE`.
