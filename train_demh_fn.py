from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
import numpy as np
from torch.utils.data import DataLoader
from models import DualGridMLMMelHarm
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse
from train_utils import train_with_curriculum

def train_demh(
        train_dataset,
        trainloader,
        valloader,
        tokenizer,
        exponent,
        subfolder,
        device_name,
        epochs=20,
        lr=1e-4,
        validations_per_epoch=10,
        tqdm_position=0
    ):

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

    model = DualGridMLMMelHarm(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=512,
        nhead=4,
        num_layers_mel=4,
        num_layers_harm=4,
        melody_length=80,
        harmony_length=80,
        pianoroll_dim=tokenizer.pianoroll_dim,
        device=device,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    curriculum_type = 'f2f'

    # save results
    os.makedirs('results/DE/', exist_ok=True)
    os.makedirs('results/DE/' + subfolder + '/', exist_ok=True)
    results_path = 'results/DE/' + subfolder + '/' + curriculum_type + str(exponent) + '.csv'

    os.makedirs('saved_models/DE/', exist_ok=True)
    os.makedirs('saved_models/DE/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/DE/' + subfolder + '/'
    transformer_path = save_dir + curriculum_type + str(exponent) + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        exponent=exponent,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id,
        validations_per_epoch=validations_per_epoch,
        tqdm_position=tqdm_position
    )