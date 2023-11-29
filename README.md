# Beyond Last Layer: Deep Features Loss function SCA
This is the repository for `Beyond Last Layer: Deep Feature Loss function in Side-Channel Analysis`


1. `run.py` will execute `random_dnn.py` for different scenarios to train and evaluate the guessing entropy, GE. 
2. test_performance.py will execute test_best_models.py for running the architecture with the best guessing entropy GE, 10 times to obtain the median.


Remarks: soft_nn and center_loss uses categorical cross-entropy (CCE) as the baseline loss function, while flr_soft_nn and flr_center_loss uses focal loss ratio (FLR) as their baseline loss function instead.


cite us at 
```
@inproceedings{yap2023beyond,
  title={Beyond the Last Layer: Deep Feature Loss Functions in Side-channel Analysis},
  author={Yap, Trevor and Picek, Stjepan and Bhasin, Shivam},
  booktitle={Proceedings of the 2023 Workshop on Attacks and Solutions in Hardware Security},
  pages={73--82},
  year={2023}
}
```
