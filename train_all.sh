#!/bin/bash

# List of Python scripts with their respective arguments

# scripts=(
#     "train_demh.py -x 5 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 16"
#     "train_demh.py -x 7 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 16"
#     "train_demh.py -x 10 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 16"
#     "train_semh.py -x 5 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 16"
#     "train_semh.py -x 7 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 16"
#     "train_semh.py -x 10 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 16"
# )

# scripts=(
#     "train_semh.py -x 4 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 32"
#     "train_semh.py -x 5 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 32"
#     "train_semh.py -x 6 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 32"
#     "train_semh.py -x 7 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 32"
#     "train_semh.py -x 9 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 32"
#     "train_semh.py -x 13 -f CA -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 32"
# )

# base2 -> Q16 vs Q4 - bar vs no bar. 1) Is Q4 better or equal to Q16? 2) Is bar better than condition?
#
# base2 Q4 bar PC only vs random Q4 bar PC only. 3) Is PC only better or equal to PC+FR?
# 4) Is base2 better than random in the new setup? 5) What do attention maps show regarding how random is trained?
#
# f2f Q4 bar PC only. 6) Is the much simpler setup equally good better (no stage, not trainable pos_emb, curriculum that starts with cross attention)?
scripts=(
    # "train_semh.py -c base2 -f Q16_L256_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c base2 -f Q16_L272_bar_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c base2 -f Q4_L64_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c base2 -f Q4_L80_bar_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c random -f Q4_L80_bar_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c base2 -f Q4_L80_bar_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 1 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c f2f -f Q4_L80_bar_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
    # ARTIFICIAL
#     "train_art_semh.py -c base2 -f Q4_L80_bar_PC -d /media/maindisk/data/synthetic_CA_train -v /media/maindisk/data/synthetic_CA_test -g 0 -e 200 -l 1e-4 -b 8"
#     "train_art_semh.py -c random -f Q4_L80_bar_PC -d /media/maindisk/data/synthetic_CA_train -v /media/maindisk/data/synthetic_CA_test -g 0 -e 200 -l 1e-4 -b 8"
#     "train_art_semh.py -c f2f -f Q4_L80_bar_PC -d /media/maindisk/data/synthetic_CA_train -v /media/maindisk/data/synthetic_CA_test -g 0 -e 200 -l 1e-4 -b 8"
    # SECOND BATCH
    # "train_semh.py -c f2f -f Q4_L64_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c f2f -f Q4_L64_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c f2f -f Q4_L80_bar_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c f2f -f Q16_L256_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 2 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c f2f -f Q16_L272_bar_PC_FR -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 2 -e 200 -l 1e-4 -b 8"
    # "train_semh.py -c f2f -f Q16_L256_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
    
    # "train_semh.py -c f2f -f Q16_L272_bar_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"

    # DE
    "train_demh.py -f Q4_L80_bar_PC -d /media/maindisk/data/hooktheory_midi_hr/CA_train -v /media/maindisk/data/hooktheory_midi_hr/CA_test -g 0 -e 200 -l 1e-4 -b 8"
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
