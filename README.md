### Environment Installation

```
conda create -n spatia python=3.9
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```


### Code Installation

```
cd model_training
pip install -e .

cd data_processing
pip install -e .
```

### Pretraining Command:

```
cd model_training/scripts
sbatch 0407_H100_run.sh
```



## Acknowledgements
Our code is based on [scPrint](https://github.com/cantinilab/scPRINT/tree/main). Thanks!
