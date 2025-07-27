# Adaptive Mixer Model

This is an implementation of TEMTP, a trajectory prediction model.

## Project Structure

```
TEMPT/
├── data/               # demo data
├── README.md           # Project documentation
├── data_loader.py      # Data loader
├── loss.py             # Loss functions
├── requirements.txt    # Environment dependency
├── tempt_config.py     # TEMPT configuration
├── tempt_train.py      # TEMPT training script
└── TEMPT.py            # TEMPT model 
```


## Usage

### Training
```bash
python tempt_train.py
```

## Dependencies

- torch
- einops
- timm
- numpy
- matplotlib