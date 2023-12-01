#!/bin/bash

# Runs all of the experiments in the paper and produces results in the figures/ directory.

mkdir -p figures

# # German Households Dataset
# python3 mean_estimation.py --name germany_consumption --eps 1
# python3 mean_estimation.py --name germany_consumption --protect_network --eps 1
# python3 mean_estimation.py --name germany_consumption --intermittent --eps 10
# python3 mean_estimation.py --name germany_consumption --intermittent --protect_network --eps 10


# # US Power Grid Dataset
# python3 mean_estimation.py --name us_power_grid --eps 0.5
# python3 mean_estimation.py --name us_power_grid --protect_network --eps 0.5
# python3 mean_estimation.py --name us_power_grid --intermittent --eps 1
# python3 mean_estimation.py --name us_power_grid --intermittent --protect_network --eps 1 

# # MSE Plots 
# python3 mean_estimation.py --name us_power_grid --task mse_plot 
# python3 mean_estimation.py --name germany_consumption --task mse_plot

# # Comparison with Rizk et al. (2023)
# python3 mean_estimation.py --name germany_consumption --protect_network --eps 1 --method rizk
# python3 mean_estimation.py --name us_power_grid --protect_network --eps 0.5 --method rizk
python3 mean_estimation.py --name us_power_grid --task mse_plot --method rizk 
python3 mean_estimation.py --name germany_consumption --task mse_plot --method rizk

# # Heterogeneous privacy budgets
# python3 mean_estimation.py --name germany_consumption --protect_network --eps 1 --distribute_budget central --eps_upper_bound 5
# python3 mean_estimation.py --name germany_consumption --intermittent --protect_network --eps 10 --distribute_budget central --eps_upper_bound 50
# python3 mean_estimation.py --name germany_consumption --intermittent --protect_network --eps 10 --distribute_budget local --eps_upper_bound 50

# python3 mean_estimation.py --name germany_consumption --task mse_plot --distribute_budget central
# python3 mean_estimation.py --name germany_consumption --task mse_plot --distribute_budget local
