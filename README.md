# Neural operators for predictor feedback control of nonlinear systems

The source code for the paper titled _Neural Operators for Predictor Feedback Control of Nonlinear Systems_

## Environment

We mainly rely on [pytorch](https://pytorch.org/).
Please first install Pytorch by following the instructions on the website.

Besides, the following packages are required and installed by
```
pip install tqdm neuraloperator torch_harmonics
```

## Reproduce our result

### Model training

Run the following file with different model architectures to reproduce the training procedure.

```shell
python train.py -model_name FNO 
```
Available architectures are FNO, DeepONet, GRU, LSTM, FNO+GRU, and DeepONet+GRU.

This will first generate the data by numerical successive approximation and then train the model.
The models' weights are saved as '<system>.pth' The weights are already available in this repository, and you may skip this step.

To save time, the data generation is turned off by default in our script.
You can download the pkl files of numerical simulation results directly via
this [link](https://drive.google.com/drive/folders/15C2AIQwt9kxbp5cUBm_CtYvxdMWUn5LI?usp=sharing) and put them to
directories named 'data/Baxter/'.

### Model evaluation

After saving the model, you can evaluate the model and reproduce the figures presented in the paper by

```shell
python evaluate.py -model_name FNO
```
This will plot 25 figures for different initial conditions.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
