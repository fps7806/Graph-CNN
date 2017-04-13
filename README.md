# Graph-CNN
Implementation of "Robust Spatial Filtering with Graph Convolutional Neural Networks" - https://arxiv.org/abs/1703.00792.
This project is still under development. Email Felipe at fps7806@rit.edu if you have any suggestions or problems.
## Requirements:
* Tensorflow >1.0
* Internet Connection (if automatically downloading datasets)

## Instructions
To obtain results on NCI1 and Cora run the following commands:

`python run_nci1.py`

`python run_cora.py`

The scripts will automatically download the datasets and run the experiments.

Note that the results will not be exactly the same every run.

Scripts will print useful information if the `self.silent = True` line is removed.

Checkpoints and Summaries that can be used with Tensorboard will be generated if the line `self.debug = True` is removed.

## NCI1 - Results

| Model        | File | Accuracy           |
| ------------- |:------------- |:-------------:|
| 64-64-GEP(32)-32-GEP(8)-FC256-FC2 |   run_nci1.py   | 84.45 +- 0.94 |

## Cora - Results

| Model        | File | Accuracy           |
| ------------- |:------------- |:-------------:|
| Embed256-48-Embed32-48-Embed32-7 |   run_cora.py   | 89.18 +- 1.96 |

## Citation

Please cite us in your publications if it helps your research:

    @article{graphcnn2017,
      Author = {Petroski Such, Felipe, and Sah, Shagan, and Dominguez, Miguel, and Pillai, Suhas, and Zhang, Chao, and Michael,  Andrew, and Cahill, Nathan, and Ptucha, Raymond},
      Journal = {arXiv preprint arXiv:1703.00792},
      Title = {Robust Spatial Filtering with Graph Convolutional Neural Networks},
      Year = {2017}
    }