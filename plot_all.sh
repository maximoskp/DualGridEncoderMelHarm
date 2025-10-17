#!/bin/bash

# List of Python scripts with their respective arguments

scripts=(
    # "plot_attn_test.py -m DE -c f2f -f Q4_L80_bar_PC -g 1 -r random"
    "plot_attn_test.py -m SE -c f2f -f Q4_L80_bar_PC -g 1 -r random"
    "plot_attn_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g 1 -r random"
    # "plot_attn_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g 1 -r random"
    "plot_attn_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g 1 -r random"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done
