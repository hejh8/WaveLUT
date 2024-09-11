# FourierDiff
Official implement of [Optimizing 4D Lookup Table for Low-light Video Enhancement via Wavelet Priori]()

## Installation
### Environment
```
conda env create --file environment.yml
conda activate WaveLUT
```
## Datasets Download
SDSD and SMID datasets [Wang](https://github.com/dvlab-research/SDSD). DID datasets [Fu](https://github.com/ciki000/DID)

### Pre-Trained Models
download this [model]() and put it into `WaveLUT/ckpt/`.

### Quick Start

```
 python main.py --config llve.yml --path_y test -i output
```
## TODO
- [x] Pre-Trained Models
- [x] Test
- [ ] Train
## Citation
```

```
