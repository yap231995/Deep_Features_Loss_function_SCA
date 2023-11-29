# Beyond Last Layer: Deep Features Loss function SCA
This is the repository for `Beyond Last Layer: Deep Feature Loss function in Side-Channel Analysis`


1. `run.py` will execute `random_dnn.py` for different scenarios to train and evaluate the guessing entropy, GE. 
2. test_performance.py will execute test_best_models.py for running the architecture with the best guessing entropy GE, 10 times to obtain the median.


Remarks: soft_nn and center_loss uses categorical cross-entropy (CCE) as the baseline loss function, while flr_soft_nn and flr_center_loss uses focal loss ratio (FLR) as their baseline loss function instead.


cite us at 
```
@inproceedings{10.1145/3605769.3623996,
author = {Yap, Trevor and Picek, Stjepan and Bhasin, Shivam},
title = {Beyond the Last Layer: Deep Feature Loss Functions in Side-Channel Analysis},
year = {2023},
isbn = {9798400702624},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3605769.3623996},
doi = {10.1145/3605769.3623996},
booktitle = {Proceedings of the 2023 Workshop on Attacks and Solutions in Hardware Security},
pages = {73–82},
numpages = {10},
keywords = {profiling attack, side-channel analysis, loss function, deep features, neural network, deep learning},
location = {<conf-loc>, <city>Copenhagen</city>, <country>Denmark</country>, </conf-loc>},
series = {ASHES '23}
}
```
