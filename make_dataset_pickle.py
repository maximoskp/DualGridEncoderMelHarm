from data_utils import CSGridMLMDataset
from GridMLM_tokenizers import CSGridMLMTokenizer

# train_dir = '/media/maindisk/data/hooktheory_midi_hr/all12_train'

# tokenizer = CSGridMLMTokenizer(
#     fixed_length=80,
#     quantization='4th',
#     intertwine_bar_info=True,
#     trim_start=False,
#     use_pc_roll=True,
#     use_full_range_melody=False
# )
# train_dataset = CSGridMLMDataset(train_dir, tokenizer, name_suffix='DE')

# for constructing all dataset
# train_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_train'
# val_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_test'
# extra_dir = '/media/maindisk/data/gjt_melodies/gjt_CA'
synthetic_train = '/media/maindisk/data/synthetic_CA_train'
synthetic_test = '/media/maindisk/data/synthetic_CA_test'

# for fixed_length, quantization, intertwine_bar_info in \
#     [(80, '4th', True), (320, '16th', True), (64, '4th', False), (256, '16th', False)]:
#     for use_pc_roll, use_full_range_melody in [(True, False), (False, True), (True, True)]:
for fixed_length, quantization, intertwine_bar_info in \
    [(80, '4th', True)]:
    for use_pc_roll, use_full_range_melody in [(True, False)]:
        tokenizer = CSGridMLMTokenizer(
            fixed_length=fixed_length,
            quantization=quantization,
            intertwine_bar_info=intertwine_bar_info,
            trim_start=False,
            use_pc_roll=use_pc_roll,
            use_full_range_melody=use_full_range_melody
        )
        suffix = f'Q{quantization[:-2]}' + f'_L{fixed_length}'
        if intertwine_bar_info:
            suffix += '_bar'
        if use_pc_roll:
            suffix += '_PC'
        if use_full_range_melody:
            suffix += '_FR'
        print(f'Processing dataset with suffix: {suffix}')
        # train_dataset = CSGridMLMDataset(train_dir, tokenizer, name_suffix=suffix)
        # val_dataset = CSGridMLMDataset(val_dir, tokenizer, name_suffix=suffix)
        # extra_dataset = CSGridMLMDataset(extra_dir, tokenizer, name_suffix=suffix)
        train_dataset = CSGridMLMDataset(synthetic_train, tokenizer, name_suffix=suffix)
        val_dataset = CSGridMLMDataset(synthetic_test, tokenizer, name_suffix=suffix)