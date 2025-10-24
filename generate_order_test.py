from generate_utils import generate_files_with_nucleus, generate_files_with_base2, generate_files_with_random, load_SE_Modular, load_DE_model
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import argparse

curriculum_types = ['random', 'base2', 'f2f']
unmasking_orders = ['random', 'start', 'end', 'certain', 'uncertain']

def main():
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for inference with a specific setup.')
    # Define arguments
    parser.add_argument('-m', '--model', type=str, help='Specify model, SE or DE', required=True)
    parser.add_argument('-c', '--curriculum', type=str, help='Specify the curriculum type name among: ' + repr(curriculum_types), required=True)
    parser.add_argument('-f', '--subfolder', type=str, help='Specify subfolder to save the model and results. This name also defines tokenizer and token setup.', required=True)
    parser.add_argument('-g', '--generation', type=str, help='Specify generation function.', required=True)
    parser.add_argument('-u', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-v', '--nvis', type=int, help='Specify number of visible tokens. Default is None.', required=False)
    parser.add_argument('-r', '--order', type=str, help='Specify unmasking order among: ' + repr(unmasking_orders) + '. Default is start.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    model_type = args.model
    curriculum_type = args.curriculum
    subfolder = ''
    if args.subfolder:
        subfolder = args.subfolder
    generation_function_name = args.generation
    nvis = None
    if args.nvis is not None:
        nvis = args.nvis
    unmasking_order = 'start' # in ['random', 'start', 'end', 'certain', 'uncertain']
    if args.order is not None:
        unmasking_order = args.order
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)

    gen_funs = {
        'base2': generate_files_with_base2,
        'random': generate_files_with_random,
        'nucleus': generate_files_with_nucleus,
    }
    gen_fun = gen_funs[generation_function_name]
    
    total_stages = None if curriculum_type == 'f2f' else 10
    condition_dim = None if 'bar' in subfolder else 16
    trainable_pos_emb = False if curriculum_type == 'f2f' else True

    grid_lenght = int(subfolder.split('_L')[1].split('_')[0])

    intertwine_bar_info = 'bar' in subfolder

    base_folder = 'MIDIs_order/testsuper/testset/'

    tokenizer = CSGridMLMTokenizer(
        fixed_length=grid_lenght,
        quantization='16th' if 'Q16' in subfolder else '4th',
        intertwine_bar_info=intertwine_bar_info,
        trim_start=False,
        use_pc_roll='PC' in subfolder,
        use_full_range_melody='FR' in subfolder
    )
    
    val_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_test'

    data_files = []
    for dirpath, _, filenames in os.walk(val_dir):
        for file in filenames:
            if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') \
                or file.endswith('.mid') or file.endswith('.midi'):
                full_path = os.path.join(dirpath, file)
                data_files.append(full_path)
    print(len(data_files))

    if 'SE' in model_type:
        model = load_SE_Modular(
            d_model=512, 
            nhead=8,
            num_layers=8,
            curriculum_type=curriculum_type,
            subfolder=subfolder,
            device_name=device_name,
            tokenizer=tokenizer,
            grid_length=grid_lenght,
            condition_dim=condition_dim,  # if not None, add a condition token of this dim at start
            unmasking_stages=total_stages,  # if not None, use stage-based unmasking
            trainable_pos_emb=trainable_pos_emb,
            nvis=nvis,
            version=model_type
        )
    else:
        model = load_DE_model(
        d_model=512, 
        nhead=8, 
        num_layers_mel=8,
        num_layers_harm=8,
        curriculum_type=curriculum_type,
        subfolder=subfolder,
        device_name=device_name,
        tokenizer=tokenizer,
        melody_length=grid_lenght,
        harmony_length=grid_lenght,
        nvis=nvis,
        version=model_type
    )

    # then create gen
    midi_folder = base_folder + model_type + '_' + unmasking_order
    if nvis is not None:
        midi_folder += '_nvis' + str(nvis)
    midi_folder += '/'
    os.makedirs(midi_folder, exist_ok=True)
    for val_idx in tqdm(range(len(data_files))):
        input_f = data_files[val_idx]
        gen_harm, real_harm, gen_score, real_score = gen_fun(
            model=model,
            tokenizer=tokenizer,
            input_f=input_f,
            mxl_folder=None,
            midi_folder=midi_folder,
            name_suffix=str(val_idx),
            use_constraints=False,
            intertwine_bar_info=intertwine_bar_info,
            normalize_tonality=False,
            temperature=0.2,
            p=0.9,
            unmasking_order=unmasking_order,
            num_stages=10,
            use_conditions=condition_dim is not None,
            create_gen=True,
            create_real=False
        )

if __name__ == '__main__':
    main()