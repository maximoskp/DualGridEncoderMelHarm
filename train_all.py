from train_demh_fn import train_demh
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import CSGridMLMTokenizer
import multiprocessing

# train_dir = '/media/maindisk/data/hooktheory/hooktheory_train'
# val_dir = '/media/maindisk/data/hooktheory/hooktheory_test'
# train_dir = '/media/maindisk/data/gjt_melodies/gjt'
# val_dir = '/media/maindisk/data/gjt_melodies/gjt'

# TODO: implement argument forwarding of unfold=True/False in models.py
subfolder = 'all12'
epochs = 20
validations_per_epoch = 10
train_dir = '/media/maindisk/data/hooktheory_midi_hr/all12_train'
val_dir = '/media/maindisk/data/hooktheory_midi_hr/CA_test'

# subfolder = 'unf_all12'
# epochs = 5
# validations_per_epoch = 10
# train_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_all12_train'
# val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_all12_test'

batchsize = 16

tokenizer = None

train_dataset = None
val_dataset = None

trainloader = None
valloader = None

def init_worker(td, vd, tl, vl, tok):
    global train_dataset, val_dataset, trainloader, valloader, tokenizer
    train_dataset = td
    val_dataset = vd
    trainloader = tl
    valloader = vl
    tokenizer = tok
# end init_worker

def train_wrapper(kwargs):
    return train_demh(
        train_dataset=train_dataset,
        trainloader=trainloader,
        valloader=valloader,
        tokenizer=tokenizer,
        **kwargs
    )
# end train_wrapper

if __name__ == "__main__":
    # Load heavy objects ONCE
    tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

    train_dataset = CSGridMLMDataset(train_dir, tokenizer, name_suffix='DE')
    val_dataset = CSGridMLMDataset(val_dir, tokenizer, name_suffix='DE')

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=CSGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=CSGridMLM_collate_fn)

    task_args = [
        {
            'exponent': 5,
            'subfolder': subfolder,
            'device_name': 'cuda:0',
            'epochs': epochs,
            'lr': 1e-4,
            'batchsize': batchsize,
            'validations_per_epoch': validations_per_epoch,
            'tqdm_position': 0
        },
        {
            'exponent': 4,
            'subfolder': subfolder,
            'device_name': 'cuda:1',
            'epochs': epochs,
            'lr': 1e-4,
            'batchsize': batchsize,
            'validations_per_epoch': validations_per_epoch,
            'tqdm_position': 1
        },
        {
            'exponent': 6,
            'subfolder': subfolder,
            'device_name': 'cuda:1',
            'epochs': epochs,
            'lr': 1e-4,
            'batchsize': batchsize,
            'validations_per_epoch': validations_per_epoch,
            'tqdm_position': 2
        },
    ]

    # Use "fork" for memory-efficient sharing (if on Unix)
    with multiprocessing.get_context("fork").Pool(
        processes=len(task_args),
        initializer=init_worker,
        initargs=(train_dataset, val_dataset, trainloader, valloader, tokenizer)
    ) as pool:
        results = pool.map(train_wrapper, task_args)

    print("All finished:", results)
# end main