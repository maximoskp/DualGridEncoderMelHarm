from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
import numpy as np
from torch.utils.data import DataLoader
from models import SEModular
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse
from train_utils import train_with_curriculum

curriculum_types = ['random', 'base2']

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a GridMLM model with a specific curriculum type.')

    # Define arguments
    parser.add_argument('-c', '--curriculum', type=str, help='Specify the curriculum type name among: ' + repr(curriculum_types), required=True)
    parser.add_argument('-f', '--subfolder', type=str, help='Specify subfolder to save the model and results. This name also defines tokenizer and token setup.', required=True)
    parser.add_argument('-d', '--datatrain', type=str, help='Specify the full path to the root folder of the training xml/mxl files', required=True)
    parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 5e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 8.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    curriculum_type = args.curriculum
    exponent = 5
    subfolder = ''
    if args.subfolder:
        subfolder = args.subfolder
    train_dir = args.datatrain
    val_dir = args.dataval
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 200
    if args.epochs:
        epochs = args.epochs
    lr = 1e-4
    if args.learningrate:
        lr = args.learningrate
    batchsize = 16
    if args.batchsize:
        batchsize = args.batchsize
    
    total_stages = None if curriculum_type == 'f2f' else 10
    condition_dim = None if 'bar' in subfolder else 16
    trainable_pos_emb = False if curriculum_type == 'f2f' else True
    
    grid_lenght = int(subfolder.split('_L')[1].split('_')[0])
    tokenizer = CSGridMLMTokenizer(
        fixed_length=grid_lenght,
        quantization='16th' if 'Q16' in subfolder else '4th',
        intertwine_bar_info='bar' in subfolder,
        trim_start=False,
        use_pc_roll='PC' in subfolder,
        use_full_range_melody='FR' in subfolder
    )

    def compute_class_weights_from_dataset(dataset, tokenizer, scheme="temp", alpha=0.5, beta=0.999, ignore_index=-100):
        """
        Compute class weights for CrossEntropyLoss given a dataset of chord tokens.
        
        Args:
            dataset: dataset loaded from pickle
            tokenizer: tokenizer object with vocab (for num_classes)
            scheme: one of ["inv", "inv_sqrt", "cb", "temp"]
            alpha: used in temperature-scaled scheme
            beta: used in class-balanced scheme
            ignore_index: value used to skip padding/masked tokens
        
        Returns:
            torch.Tensor of shape [num_classes] with normalized class weights
        """
        num_classes = len(tokenizer.vocab)
        counts = torch.zeros(num_classes, dtype=torch.float)

        # Count occurrences of chord tokens across dataset
        for item in dataset:
            tokens = torch.tensor(item["harmony_ids"], dtype=torch.long)
            tokens = tokens[tokens != ignore_index]  # filter padding/masked tokens
            counts += torch.bincount(tokens, minlength=num_classes).float()

        freqs = counts / counts.sum()

        # Apply weighting scheme
        if scheme == "inv":  # 1/p
            weights = 1.0 / (freqs + 1e-9)

        elif scheme == "inv_sqrt":  # 1/sqrt(p)
            weights = 1.0 / torch.sqrt(freqs + 1e-9)

        elif scheme == "cb":  # Class-balanced loss
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-9)

        elif scheme == "temp":  # temperature-scaled: p^-alpha
            weights = (freqs + 1e-9) ** (-alpha)

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Normalize so average weight = 1
        weights = weights / weights.mean()

        return weights
    # end compute_class_weights_from_dataset

    train_dataset = CSGridMLMDataset(train_dir, tokenizer, name_suffix=subfolder)
    val_dataset = CSGridMLMDataset(val_dir, tokenizer, name_suffix=subfolder)

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=CSGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=CSGridMLM_collate_fn)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection
    
    loss_fn=CrossEntropyLoss(ignore_index=-100)
    # # Precompute once before training
    # class_weights = compute_class_weights_from_dataset(
    #     train_dataset, tokenizer, scheme="temp", alpha=0.5
    # )

    # # Define loss function with weights
    # loss_fn = torch.nn.CrossEntropyLoss(
    #     weight=class_weights.to(device), ignore_index=-100
    # )
    model = SEModular(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=512,
        nhead=8,
        num_layers=8,
        grid_length=grid_lenght,
        pianoroll_dim=tokenizer.pianoroll_dim,
        condition_dim=condition_dim,  # if not None, add a condition token of this dim at start
        unmasking_stages=total_stages,  # if not None, use stage-based unmasking
        trainable_pos_emb=trainable_pos_emb,
        device=device,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/SE/', exist_ok=True)
    os.makedirs('results/SE/' + subfolder + '/', exist_ok=True)
    results_path = 'results/SE/' + subfolder + '/' + curriculum_type + '.csv'

    os.makedirs('saved_models/SE/', exist_ok=True)
    os.makedirs('saved_models/SE/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/SE/' + subfolder + '/'
    transformer_path = save_dir + curriculum_type + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        curriculum_type=curriculum_type,
        epochs=epochs,
        condition_dim=condition_dim,
        exponent=exponent,
        total_stages=total_stages,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id
    )
    
# end main

if __name__ == '__main__':
    main()