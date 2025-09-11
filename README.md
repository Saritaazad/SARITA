# SARITA: Hourly Rainfall Forecasting

This repository contains code for training and evaluating a DConvLSTM model that incorporates attention based spatial correlation (ASAC) for hourly precipitation forecasting.

## 📦 Setup Instructions

### 1. Install Environment

Make sure you have [Conda](https://docs.conda.io/en/latest/) installed. Then run:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

Replace `<env_name>` with the name specified in the `environment.yml` file.

---

### 2. Prepare Dataset

Organize your dataset folder as follows:

```
datasets/
├── train/
│   ├── weather.csv
│   └── moran.csv
├── val/
│   ├── weather.csv
│   └── moran.csv
└── test/
    ├── weather.csv
    └── moran.csv
```

Ensure that:
- `weather.csv` contains normalized precipitation data.
- `moran.csv` contains the corresponding spatial autocorrelation matrices.

---

### 3. Modify Paths in `run.py`

Open the `run.py` file and edit the following arguments to point to your dataset:

```python
parser.add_argument('--train_data_paths', type=str, default='path/to/train/weather.csv,path/to/train/moran.csv')
parser.add_argument('--valid_data_paths', type=str, default='path/to/val/weather.csv,path/to/val/moran.csv')
parser.add_argument('--test_data_paths', type=str, default='path/to/test/weather.csv,path/to/test/moran.csv')
```

Replace these with the actual paths on your system.

---

### 4. Run the Model

Once above setup is complete, if testing provide model path in
```python
parser.add_argument('--pretrained_model', type=str, default='save/model.ckpt-5')
```
otherwise comment it out, finally you can train or test the model by running:
```bash
python run.py --is_training 1   # for training
python run.py --is_training 0   # for testing
```

Model checkpoints, predictions, and loss plots will be saved in the specified `save` and `predict` directories.

---

## 🔧 Notes
- Ensure `CUDA` is available if using GPU acceleration.
- If using `wandb`, make sure you’re logged in via `wandb login`.

---
