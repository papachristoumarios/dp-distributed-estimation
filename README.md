# dp-distributed-estimation

Supplementary code for the paper _Differentially Private Distributed Estimation and Learning_. 

## Installation and Usage

To install the required packages use `pip install requirements.txt`. To run the distributed estimation algorithms use

```bash
    python mean_estimation.py --name {name} --eps 1 --delta 0.01 
```

The following datasets are available:

 * `us_power_grid`: US Power Grid Network
 * `germany_consumption`: German Households Consumption Dataset
 * `ieee_33_bus`: IEEE 33 Bus network

For more options run `python mean_estimation.py --help`. 


## Reproducing Experiments

To reproduce the experiments of the paper run the `run_all_experiments.sh` script as 

```bash
bash run_all_experiments.sh
```

The resulting figures will be saved in the `figures` directory. 