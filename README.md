# Graph Neural Network Guided Local Search for the Traveling Salesperson Problem

Code accompanying the paper [Graph Neural Network Guided Local Search for the Traveling Salesperson Problem](https://arxiv.org/abs/2110.05291).

Want to skip straight to the [example](https://github.com/ben-hudson/gnngls#example)?

## Setup
Requires [pipenv](https://pipenv.pypa.io).

Run `pipenv install` in the root directory. Then, run `pipenv shell` to activate the environment.

## Datasets
We have separated dataset generation into two parts: instance generation and preprocessing. You can generate solved TSP instances using:
```
./generate_instances.py <number of instances to generate> <number of nodes> <dataset directory>
```

The specified directory is created. Each instance is a pickled `networkx.Graph`.

Then, prepare the dataset using:
```
./preprocess_dataset.py <dataset directory>
```
This splits the dataset into training, validation, and test sets written to `train.txt`, `val.txt`, and `test.txt` respectively. It also fits a scaler to the training set.

After this step, the datasets can be easily manipulated using `gnngls.TSPDataset`. For example, in [train.py](https://github.com/ben-hudson/gnngls/blob/master/scripts/train.py#L89).

## Training
Train the model using:
```
./train.py <dataset directory> <tensorboard directory> --use_gpu
```
The default optional arguments are those used in the paper. A new directory will be created under the specified Tensorboard directory, and checkpoints and training progress will be written there.

## Testing
Evaluate the model using:
```
./test.py <dataset directory>/test.txt <checkpoint path> <run directory> regret_pred --use_gpu
```
The default optional arguments are those used in the paper. The search progress for all instances in the dataset will be written to the specified run directory as a pickled `pandas.DataFrame`.

## Example
The following is a simple demonstration to help you get started 🙂
```
pipenv install
pipenv shell
cd scripts
python generate_instances.py 10 500 data
python preprocess_dataset.py data --n_train 400 --n_val 50 --n_test 50
python train.py data models --use_gpu
python test.py data/test.txt models/<new model directory>/checkpoint_best_val.pt runs regret_pred --use_gpu
```

## Citation
If you this code is useful in your research, please cite our paper:
```
@inproceedings{hudson2022gnngls,
  title = {Graph Neural Network Guided Local Search for the Traveling Salesperson Problem},
  author = {Hudson, Benjamin and Li, Qingbiao and Malencia, Matthew and Prorok, Amanda},
  booktitle = {International Conference on Learning Representations},
  year = {2022},
}
```
