# Neural operators for predictor feedback control of nonlinear systems

The source code for the paper titled _Neural operators for predictor feedback control of nonlinear systems_

## Environment

We mainly rely on [pytorch](https://pytorch.org/).
Please first install pytorch following the instruction from the website.

Besides, two packages are required.

```
pip install tqdm neuraloperator
```

## Reproduce our result

### Model training

Run the following two files to reproduce the training procedure.

```shell
python -s Baxter train.py
python -s Unicycle train.py
```

This will first generation the data by numerical successive approximation, and train the model.

The model will be saved as '<system>.pth'.

The model is available in the repository, and you can skip this step.

To save time, the data generation is turned off by default in our script.
You can download the pkl files of numerical simulation results directly via
this [link](https://drive.google.com/drive/folders/15C2AIQwt9kxbp5cUBm_CtYvxdMWUn5LI?usp=sharing) and put them to
directories named 'data/\<system\>/'.

### Model evaluation

After saving the model, you can evaluate the model and reproduce the figures presented in the paper by

```shell
python evaluate.py -s Baxter
python evaluate.py -s Unicycle
```

This will plot 25 figures for different initial conditions.
The first figures for each system are presented in our paper.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
