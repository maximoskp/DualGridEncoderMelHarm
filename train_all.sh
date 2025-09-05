#!/bin/bash

# List of Python scripts with their respective arguments

scripts=(
    "train_demh.py -c random -s 10 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 100 -l 1e-5 -b 8"
    "train_demh.py -c random -s 20 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 100 -l 1e-5 -b 8"
    "train_demh.py -c random -s 80 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 100 -l 1e-5 -b 8"
    "train_demh.py -c step -s 80 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 100 -l 1e-5 -b 8"
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
