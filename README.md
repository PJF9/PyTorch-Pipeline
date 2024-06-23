# PyTorch-Pipeline
This repository contains a comprehensive pipeline for training and evaluating deep learning models using the PyTorch framework. The pipeline is designed specifically for time-series binary classification tasks but can be modified for other use cases.

---------------

## Using this Pipeline
Follow these steps to use the pipeline effectively. The steps guide you through creating a custom dataset, defining a model, and configuring the necessary scripts.

### Step 1: Create Your Dataset
1. Navigate to `src.dataset.Datasets`
2. Update the `YourDataset` class to fit your specific problem. For detailed guidance, refer to the [PyTorch data tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

### Step 2: Create Your Model
1. Navigate to `src.models.baselines`.
2. Modify the `TransformerEncoderAll` class to define your model architecture, or create a new model for scratch.

### Step 3: Configure the Scripts
1. Dataset Configuration:
    * Update the dataset name in `src.dataset.__init__.py`, `train.py`, `evaluate.py`, and `hyperparameter_tuning.py`.

2. Model Configuration:
    * Update the model name in `src.models.__init__.py`, `train.py`, `evaluate.py`, and `hyperparameter_tuning.py`.

3. Hyperparameter Tuning Script:
    * Configure `hyperparameter_tuning.py` by setting the following global variables such as **DATASET_PATH**, **INPUT_SHAPE**, **WINDOW**, **GAMMA**, **EPOCHS**, and **TRIALS**.
    * Adjust the suggestions on the `objective` function as needed.

4. Training Script:
    * In `train.py`, modify the **config** and **model_kwards** dictionaries.
    * The current setup trains a model on list of multiple datasets. To train on a single dataset, refer to `hyperparameter_tuning.py` and adjust accordingly.

5. Evaluation Script:
    * In `evaluate.py`, update the **config** and **model_kwards** dictionaries.
    * Ensure the correct path and class for the pre-trained model are specified when loading it.

---------------

## Additional Notes
* Make sure to have all the dependencies installed before running the scripts.
* Follow best practices for data preprocessing and model evaluation to achieve optimal results.
* Contributions and feedback are welcome. Please feel free to open issues or submit pull requests.

-

Happy training!
