from generate_utils import generate_files_with_beam, load_DE_model, load_SE_model
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm

device_name = 'cuda:1'

tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

val_dir = '/media/maindisk/data/gjt_melodies/gjt_CA'

data_files = []
for dirpath, _, filenames in os.walk(val_dir):
    for file in filenames:
        if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') \
            or file.endswith('.mid') or file.endswith('.midi'):
            full_path = os.path.join(dirpath, file)
            data_files.append(full_path)
print(len(data_files))

curriculum_type = 'f2f'

exponent, nvis = 5, None
# exponent, nvis = 5, 0
# exponent, nvis = 5, 5
# exponent, nvis = 5, 15
# exponent, nvis = 5, 31
# exponent, nvis = 5, 51

# exponent, nvis = 7, None
# exponent, nvis = 7, 0
# exponent, nvis = 7, 5
# exponent, nvis = 7, 15
# exponent, nvis = 7, 31
# exponent, nvis = 7, 51

# exponent, nvis = 10, None
# exponent, nvis = 10, 0
# exponent, nvis = 10, 5
# exponent, nvis = 10, 15
# exponent, nvis = 10, 31
# exponent, nvis = 10, 50

unmasking_order = 'certain' # in ['random', 'start', 'end', 'certain', 'uncertain']

subfolder = 'DE/CA'

model = load_DE_model(
    d_model=512, 
    nhead=4, 
    num_layers_mel=4,
    num_layers_harm=4,
    curriculum_type=curriculum_type,
    subfolder=subfolder,
    device_name=device_name,
    tokenizer=tokenizer,
    melody_length=80,
    harmony_length=80,
    exponent=exponent,
    nvis=nvis
)

# then create gen
for exponent in [5,7,10]:
    for unmasking_order in ['random', 'start', 'end', 'certain', 'uncertain']:
        print('creating gen', subfolder, exponent, unmasking_order)
        midi_folder = 'MIDIs_jazz/' + subfolder + '/' + curriculum_type + str(exponent)
        if nvis is not None:
            midi_folder += '_nvis' + str(nvis)
        midi_folder += '_' + unmasking_order + '/'
        os.makedirs(midi_folder, exist_ok=True)

        for val_idx in tqdm(range(len(data_files))):
            input_f = data_files[val_idx]
            gen_harm, real_harm, gen_score, real_score, avg_diffs = generate_files_with_beam(
                model=model,
                tokenizer=tokenizer,
                input_f=input_f,
                mxl_folder=None,
                midi_folder=midi_folder,
                name_suffix=str(val_idx),
                intertwine_bar_info=True,
                normalize_tonality=False,
                use_constraints=False,
                temperature=1.0,
                beam_size=5,
                top_k=50,
                unmasking_order=unmasking_order, # in ['random', 'start', 'end', 'certain', 'uncertain']
                create_gen=True,
                create_real=False
            )