# WaveLUT
Official implement of [Optimizing 4D Lookup Table for Low-light Video Enhancement via Wavelet Priori]()

## Installation
### Environment
```
conda activate WaveLUT
pip install -r requirements
python models/WaveLUT/transform/setup.py install
```
## Datasets Download
SDSD and SMID datasets [Wang](https://github.com/dvlab-research/SDSD). DID datasets [Fu](https://github.com/ciki000/DID)

### Pre-Trained Models
download this [model]() and put it into `WaveLUT/ckpt/`.

### Quick Start


```
python quantitative_test.py -opt [YOUR_yml]
```

## TODO
- [x] Pre-Trained Models
- [x] Test
- [ ] Train
## Citation
```

```
