#!/bin/bash

# List of Python scripts with their respective arguments

# base2 -> Q16 vs Q4 - bar vs no bar. 1) Is Q4 better or equal to Q16? 2) Is bar better than condition?
#
# base2 Q4 bar PC only vs random Q4 bar PC only. 3) Is PC only better or equal to PC+FR?
# 4) Is base2 better than random in the new setup? 5) What do attention maps show regarding how random is trained?
#
# f2f Q4 bar PC only. 6) Is the much simpler setup equally good better (no stage, not trainable pos_emb, curriculum that starts with cross attention)?
scripts=(
    # "generate_modular_test.py -c base2 -f Q16_L256_PC_FR -g base2"
    # "generate_modular_test.py -c base2 -f Q16_L272_bar_PC_FR -g base2"
    # "generate_modular_test.py -c base2 -f Q4_L64_PC_FR -g base2"
    # "generate_modular_test.py -c base2 -f Q4_L80_bar_PC_FR -g base2"
    # "generate_modular_test.py -c random -f Q4_L80_bar_PC -g random"
    # "generate_modular_test.py -c base2 -f Q4_L80_bar_PC -g base2"
    # "generate_modular_test.py -c f2f -f Q4_L80_bar_PC  -g nucleus"
    # "generate_modular_test.py -c f2f -f Q4_L80_bar_PC  -g base2"
    # "generate_modular_test.py -c f2f -f Q4_L80_bar_PC  -g random"
    # "generate_modular_test.py -c random -f Q4_L80_bar_PC -g nucleus"
    # "generate_modular_jazz.py -c base2 -f Q16_L256_PC_FR -g base2"
    # "generate_modular_jazz.py -c base2 -f Q16_L272_bar_PC_FR -g base2"
    # "generate_modular_jazz.py -c base2 -f Q4_L64_PC_FR -g base2"
    # "generate_modular_jazz.py -c base2 -f Q4_L80_bar_PC_FR -g base2"
    # "generate_modular_jazz.py -c random -f Q4_L80_bar_PC -g random"
    # "generate_modular_jazz.py -c base2 -f Q4_L80_bar_PC -g base2"
    # "generate_modular_jazz.py -c f2f -f Q4_L80_bar_PC  -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q4_L80_bar_PC  -g base2"
    # "generate_modular_jazz.py -c f2f -f Q4_L80_bar_PC  -g random"
    # "generate_modular_jazz.py -c random -f Q4_L80_bar_PC -g nucleus"

    # "generate_modular_test.py -c f2f -f Q4_L64_PC -g nucleus"
    # "generate_modular_test.py -c f2f -f Q4_L64_PC_FR -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q4_L64_PC -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q4_L64_PC_FR -g nucleus"

    # "generate_modular_test.py -c f2f -f Q16_L256_PC -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q16_L256_PC -g nucleus"

    # "generate_modular_test.py -c f2f -f Q16_L272_bar_PC -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q16_L272_bar_PC -g nucleus"

    # "generate_modular_test.py -c f2f -f Q16_L272_bar_PC_FR -g nucleus"
    # "generate_modular_test.py -c f2f -f Q16_L256_PC_FR -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q16_L272_bar_PC_FR -g nucleus"
    # "generate_modular_jazz.py -c f2f -f Q16_L256_PC_FR -g nucleus"

    # # ORDER ==================================================================
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_test.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_jazz.py -m SE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_jazz.py -m DE -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # # DE4 =================================================================

    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r start"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r end"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r random"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r uncertain"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r start"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r end"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r random"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r certain"
    # "generate_order_test.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r uncertain"

    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_jazz.py -m DE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # # SE16 =================================================================

    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r start"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r end"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r random"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r uncertain"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r start"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r end"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r random"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r certain"
    # "generate_order_test.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r uncertain"

    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_jazz.py -m SE16 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # # SE4 =================================================================

    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r start"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r end"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r random"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r uncertain"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r start"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r end"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r random"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r certain"
    # "generate_order_test.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -v 0 -r uncertain"

    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_jazz.py -m SE4 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # # # DE_cross =================================================================

    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r start"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r end"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r random"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r certain"
    # "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -v 0 -r uncertain"

    # # # # DE_no M HM =================================================================

    # "generate_order_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r start"
    # "generate_order_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r end"
    # "generate_order_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r random"
    # "generate_order_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    # "generate_order_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r uncertain"
    # "generate_order_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r start"
    # "generate_order_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r end"
    # "generate_order_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r random"
    # "generate_order_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    # "generate_order_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r uncertain"

    # "generate_order_jazz.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"
    # "generate_order_jazz.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r start"
    # "generate_order_jazz.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r end"
    # "generate_order_jazz.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r random"
    # "generate_order_jazz.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    # "generate_order_jazz.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r uncertain"

    # # # # DE learned pos, no vis - SE no vis
    "generate_order_test.py -m DE_lp -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    "generate_order_test.py -m DE_v0 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    "generate_order_test.py -m SE_v0 -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    "generate_order_test.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    "generate_order_test.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    "generate_order_test.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 0 -r certain"
    
    "generate_order_jazz.py -m DE_lp -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    "generate_order_jazz.py -m DE_v0 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    "generate_order_jazz.py -m SE_v0 -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    "generate_order_jazz.py -m DE_no_Mself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    "generate_order_jazz.py -m DE_no_MHself -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
    "generate_order_jazz.py -m DE_cross -c f2f -f Q4_L80_bar_PC -g nucleus -u 1 -r certain"
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
