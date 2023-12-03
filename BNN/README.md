### Prerequisites
* PyTorch
* Numpy
* Matplotlib

#### Regression Experiments

I conducted both homoscedastic and heteroscedastic regression experiments on toy datasets (generated with [Gaussian Process ground truth](https://colab.research.google.com/drive/1t-OmK57w31ukbuftqk-1zAFzIgZhSMwG)) as well as real-world data from six [UCI datasets](https://archive.ics.uci.edu/ml/datasets.php). 

*Notebooks/classification/(ModelName)_(ExperimentType).ipynb*: This includes experiments using (ModelName) on (ExperimentType), covering both homoscedastic and heteroscedastic types. The heteroscedastic notebooks also feature experiments on toy and UCI datasets for each (ModelName).

Google Colab notebooks are available, enabling GPU usage for free. All dependencies and datasets are included in the notebooks, with the only requirement being to select: Runtime -> Change runtime type -> Hardware accelerator -> GPU.

#### MNIST Classification Experiments

*train_(ModelName)_(Dataset).py*: This script trains (ModelName) on (Dataset), with training metrics and model weights saved in designated directories.

*src/*: This directory contains general utilities and model definitions.

*Notebooks/classification*: A collection of notebooks for training models, evaluating them, and conducting digit rotation uncertainty experiments. These notebooks also facilitate weight distribution visualization and weight pruning, and allow loading of pre-trained models for further experimentation.

### Bayes by Backprop (BBP)
(Refer to the [BBP Paper](https://arxiv.org/abs/1505.05424))

Regression models in Colab notebooks: [BBP homoscedastic](https://colab.research.google.com/drive/1K1I_UNRFwPt9l6RRkp8IYg1504PR9q4L) / [BBP heteroscedastic](https://colab.research.google.com/drive/13oTnT6oKnB6NNBPVAczx8X-QEot2hfp9)

To train a model on MNIST:
```bash
python train_BayesByBackprop_MNIST.py [--model [MODEL]] [--prior_sig [PRIOR_SIG]] [--epochs [EPOCHS]] [--lr [LR]] [--n_samples [N_SAMPLES]] [--models_dir [MODELS_DIR]] [--results_dir [RESULTS_DIR]]

For script argument details:

python train_BayesByBackprop_MNIST.py -h
