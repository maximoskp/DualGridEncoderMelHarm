from generate_utils import nucleus_token_by_token_generate, load_SE_Modular, load_DE_model
from GridMLM_tokenizers import CSGridMLMTokenizer
from plot_utils import save_attention_maps_with_split, save_attention_maps
import torch
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
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-v', '--nvis', type=int, help='Specify number of visible tokens. Default is None.', required=False)
    parser.add_argument('-r', '--order', type=str, help='Specify unmasking order among: ' + repr(unmasking_orders) + '. Default is start.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    model_type = args.model
    curriculum_type = args.curriculum
    subfolder = ''
    if args.subfolder:
        subfolder = args.subfolder
    nvis = None
    if args.nvis is not None:
        nvis = args.nvis
    unmasking_order = 'random' # in ['random', 'start', 'end', 'certain', 'uncertain']
    if args.order is not None:
        unmasking_order = args.order
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    use_constraints = False
    use_conditions = 'bar' in subfolder
    
    total_stages = None if curriculum_type == 'f2f' else 10
    condition_dim = None if 'bar' in subfolder else 16
    trainable_pos_emb = False if curriculum_type == 'f2f' else True

    grid_lenght = int(subfolder.split('_L')[1].split('_')[0])

    intertwine_bar_info = 'bar' in subfolder

    base_folder = 'figs/avg_attn_maps/'

    tokenizer = CSGridMLMTokenizer(
        fixed_length=grid_lenght,
        quantization='16th' if 'Q16' in subfolder else '4th',
        intertwine_bar_info=intertwine_bar_info,
        trim_start=False,
        use_pc_roll='PC' in subfolder,
        use_full_range_melody='FR' in subfolder
    )
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id
    
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
    figs_folder = base_folder + model_type + '_' + unmasking_order
    if nvis is not None:
        figs_folder += '_nvis' + str(nvis)
    figs_folder += '/'
    os.makedirs(figs_folder, exist_ok=True)
    # decide if model has cross-attention and/or self-attention
    if 'SE' in model_type:
        has_self_attention = True
        has_cross_attention = False
    else:
        has_self_attention = True
        has_cross_attention = True
        if model_type == 'DE_no_MHself' or model_type == 'DE_cross':
            has_self_attention = False  # in f2f DE, no self-attention in harmony encoder
    if has_self_attention:
        total_self_attns = None
    if has_cross_attention:
        total_cross_attns = None
    for val_idx in tqdm(range(len(data_files))):
        input_f = data_files[val_idx]
        input_encoded = tokenizer.encode(
            input_f,
            keep_durations=True,
            normalize_tonality=False,
        )

        harmony_real = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
        harmony_input = torch.LongTensor(input_encoded['harmony_ids']).reshape(1, len(input_encoded['harmony_ids']))
        # if intertwine_bar_info is True and use_constraints is False, we only need to pass
        # the bar information as a constraint, not the chords, or anything else
        # so mask out everything except from bar_token_ids
        if intertwine_bar_info and not use_constraints:
            harmony_input[ harmony_input != tokenizer.bar_token_id ] = tokenizer.mask_token_id
        melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
        if use_conditions:
            conditioning_vec = torch.FloatTensor( input_encoded['time_signature'] ).reshape( 1, len(input_encoded['time_signature']) )
        else:
            conditioning_vec = None
        _ = nucleus_token_by_token_generate(
            model=model,
            melody_grid=melody_grid.to(model.device),
            mask_token_id=tokenizer.mask_token_id,
            temperature=0.2,
            pad_token_id=pad_token_id,      # token ID for <pad>
            nc_token_id=nc_token_id,       # token ID for <nc>
            force_fill=True,         # disallow <pad>/<nc> before melody ends
            chord_constraints = harmony_input.to(model.device) if use_constraints or intertwine_bar_info else None,
            p=0.9,
            unmasking_order=unmasking_order,
            num_stages=100,
            conditioning_vec=conditioning_vec
        )
        if has_self_attention:
            if has_cross_attention:
                if total_self_attns is None:
                    total_self_attns, total_cross_attns = model.get_attention_maps()
                else:
                    self_attns_tmp, cross_attns_tmp = model.get_attention_maps()
                    for layer in range(len(self_attns_tmp)):
                        for head in range(len(self_attns_tmp[layer])):
                            total_self_attns[layer][head] += self_attns_tmp[layer][head]
                    for layer in range(len(cross_attns_tmp)):
                        for head in range(len(cross_attns_tmp[layer])):
                            total_cross_attns[layer][head] += cross_attns_tmp[layer][head]
            else:
                if total_self_attns is None:
                    total_self_attns = model.get_attention_maps()
                else:
                    self_attns_tmp = model.get_attention_maps()
                    for layer in range(len(self_attns_tmp)):
                        for head in range(len(self_attns_tmp[layer])):
                            total_self_attns[layer][head] += self_attns_tmp[layer][head]
        else:
            if total_cross_attns is None:
                total_cross_attns = model.get_attention_maps()
            else:
                cross_attns_tmp = model.get_attention_maps()
                for layer in range(len(cross_attns_tmp)):
                    for head in range(len(cross_attns_tmp[layer])):
                        total_cross_attns[layer][head] += cross_attns_tmp[layer][head]
    if 'SE' in model_type:
        save_attention_maps_with_split(
            total_self_attns,
            melody_len=grid_lenght,
            save_dir=figs_folder + '/',
            prefix='self_' + model_type,
            title_info=False
        )
    else:
        if has_self_attention:
            save_attention_maps(
                total_self_attns,
                save_dir=figs_folder + '/',
                prefix='self_' + model_type,
                title_info=False
            )
        if has_cross_attention:
            save_attention_maps(
                total_cross_attns,
                save_dir=figs_folder + '/',
                prefix='cross_' + model_type,
                title_info=False
            )

if __name__ == '__main__':
    main()