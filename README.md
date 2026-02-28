# ltlgen-transformer
This project generates LTL formulas with specific distributions using a Transformer-based two-stage generation method.

# Installation
nuXmv: Download and install the nuXmv in the folder(./IJCAI24) from https://nuxmv.fbk.eu/theme/download.php?file=nuXmv-2.0.0-linux64.tar.gz

# Environment
Please use the command to get an experimental environment as follows:
```sh
conda env create -f environment.yaml
```
# Framework
```
./train.py: The script to train the LTLGeng and LTLGeni. You can select the train phase with the arguments '--phase' and change the corresponding 'token_types'. 
./test.py:  The script to generate LTLGen, LTLGeng + random, random + LTLGeni, random + random. You can select the model by '--genm' argument. 
./model:    The directory contains the model for the three industrial datasets that have been trained in two phases and named with prefix 'first' or 'second'. 
./data:     The directory contains the original dataset and the generated dataset used in the paper. 
```
# Run
For training:
```sh
python train.py --phase generation --token_types 10
python train.py --phase generation --token_types 10 --td 'data/OriginalData/amba.json' --device 5
python train.py --phase instanstiation --token_types 31
```

For testing:
```sh
python test.py --tm ./model/amba/first_step{92609}-lr{0.0001}-early{10000}-acc{0.93}.pth --tm2 ./model/amba/second_step{285636}-lr{0.0001}-early{10000}-acc{0.99}.pth --genm 1
```
