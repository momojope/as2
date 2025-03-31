#import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import random

import os
from tqdm import tqdm
import time
import argparse

from data import get_arithmetic_dataset
from lstm_solution import LSTMLM
from gpt_solution import GPT
from trainer_solution import train as train_model
from checkpointing import get_all_checkpoints_per_trials
from plotter import plot_loss_accs

########################################################################################
########################################################################################

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################################################################
########################################################################################

class DummyScheduler:
    """
    Dummy LR Scheduler that supports standard methods like state_dict, load_state_dict, etc.,
    but does nothing to the optimizer or learning rates.
    """
    def __init__(self, optimizer, *args, **kwargs):
        """
        Initialize the DummyScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer (required to match the API, not used).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.optimizer = optimizer
        self._state = {}

    def step(self, *args, **kwargs):
        """
        Dummy step function that does nothing.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def state_dict(self):
        """
        Return the state of the scheduler as a dictionary.

        Returns:
            dict: A dictionary representing the scheduler's state.
        """
        return self._state

    def load_state_dict(self, state_dict):
        """
        Load the scheduler's state from a dictionary.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        self._state.update(state_dict)

    def get_last_lr(self):
        """
        Get the last computed learning rate(s).

        Returns:
            list: A list of the last learning rates.
        """
        return [group['lr'] for group in self.optimizer.param_groups]


########################################################################################
########################################################################################

def train(args, exp_param: str=None):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)
    seed = args.seed

    # # Create a directory to save the experiment results
    # checkpoint_path = os.path.join(args.log_dir, str(args.exp_id))
    # i=0
    # while os.path.exists(checkpoint_path):
    #     i+=1
    #     checkpoint_path = os.path.join(args.log_dir, str(i))
    # os.makedirs(checkpoint_path, exist_ok=True)
    if exp_param is not None:
        args.exp_name = f"{exp_param}_seed_{seed}"
    elif exp_param is None:
        args.exp_name = f"seed_{seed}"

    checkpoint_path = os.path.join(args.log_dir, args.exp_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    ## Print parameters
    if args.verbose :
        print("=="*60)
        for k, v in vars(args).items() :
            print(k, ":", v)
        print("=="*60)

    # Data
    (train_dataset, valid_dataset), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(
        args.p, args.p, args.operator, args.r_train, args.operation_orders, is_symmetric=False, shuffle=True, seed=args.seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=min(args.train_batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
    )

    train_dataloader_for_eval = DataLoader(
        train_dataset,
        batch_size=min(args.eval_batch_size, len(train_dataset)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=min(args.eval_batch_size, len(valid_dataset)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    vocabulary_size = len(tokenizer)
    if args.model == "lstm":
        model = LSTMLM(
            vocabulary_size = vocabulary_size, 
            embedding_size = args.embedding_size, 
            hidden_size = args.hidden_size, 
            num_layers = args.num_layers, 
            dropout = args.dropout,
            padding_index = padding_index,
            bias_lstm = True,
            bias_classifier = args.bias_classifier,
            share_embeddings = args.share_embeddings
        )
    elif args.model == "gpt":
        model = GPT(
            num_heads = args.num_heads, 
            num_layers = args.num_layers,
            embedding_size = args.embedding_size,
            vocabulary_size = vocabulary_size,
            sequence_length = MAX_LENGTH,
            multiplier = 4,
            dropout = args.dropout,
            non_linearity = "gelu",
            padding_index = padding_index,
            bias_attention = True,
            bias_classifier = args.bias_classifier,
            share_embeddings = args.share_embeddings
        )
    else:
        raise ValueError("Unknown model {0}".format(args.model))

    #print(model)
    model = model.to(args.device)

    if args.verbose : 
        print("Model :", model, "\n")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model trainable parameters : {n_params}")

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # ==========================
    # Learning rate scheduler
    scheduler = DummyScheduler(optimizer) # Dummy scheduler that does nothing

    # StepLR reduces the learning rate by gamma after step_size steps
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=args.n_steps // 3,  # Reduce LR after 1/3 of total steps
    #     gamma=0.1  # Reduce LR by factor of 10
    # )
    # ==========================

    # Train    
    all_metrics = train_model(
        model, train_dataloader, train_dataloader_for_eval, valid_dataloader, optimizer, scheduler,
        args.device, 
        args.exp_name, checkpoint_path, 
        n_steps=args.n_steps,
        eval_first=args.eval_first,
        eval_period=args.eval_period,
        print_step=args.print_step,
        save_model_step=args.save_model_step,
        save_statistic_step=args.save_statistic_step,
        verbose=args.verbose
    )
    
    # Plot
    plot_loss_accs(
        all_metrics, multiple_runs=False, log_x=False, log_y=False,
        fileName=args.exp_name, filePath=checkpoint_path, show=False)

    return all_metrics, checkpoint_path


########################################################################################
########################################################################################

def train_m_models(args, exp_param: str=None, M:int=None, seeds:list=None):
    """Train M models and plot the loss and accuracies of each model separately."""
    assert M is not None or seeds is not None, "Either M or seeds should be provided."
    if seeds is not None:
        M = len(seeds)
    else :
        seeds = [args.seed + m if args.seed is not None else None for m in range(M)]
    all_checkpoint_paths = []
    for seed, m in zip(seeds, range(M)):
        print(f"Model {m+1}/{M}")
        args.exp_id = m # Set the experiment id
        args.seed = seed # Set the seed
        all_metrics, checkpoint_path = train(args, exp_param) # Train the model
        all_checkpoint_paths.append(checkpoint_path)

    all_models_per_trials, all_metrics = get_all_checkpoints_per_trials(
        all_checkpoint_paths, args.exp_name, just_files=True, verbose=args.verbose)

    # Plot
    plot_loss_accs(
        all_metrics, multiple_runs=True, log_x=False, log_y=False,
        fileName=f'{args.exp_name}_M={M}', filePath=args.log_dir, show=False)

    return all_models_per_trials, all_metrics, all_checkpoint_paths

########################################################################################
########################################################################################

class Arguments:
    # Data
    p: int = 31 # Prime number
    operator : str = "+" # ["+", "-", "*", "/"]
    r_train : float = .5
    operation_orders : int = 2 # 2, 3 or [2, 3]
    train_batch_size: int = 512
    eval_batch_size: int = 2**12
    num_workers: int = 0

    # Model
    model: str = 'lstm' # [lstm, gpt]
    num_heads: int = 4
    num_layers: int = 2
    embedding_size: int = 2**7
    hidden_size: int = 2**7
    dropout : float = 0.0
    share_embeddings : bool = False
    bias_classifier : bool = True

    # Optimization
    optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-0

    # Training
    n_steps: int = 10**4 * 1 + 1
    eval_first: int = 10**2 * 1
    eval_period: int = 10**2 * 1
    print_step: int = 10**2 * 1
    save_model_step: int = 10**3
    save_statistic_step: int = 10**3

    # Experiment & Miscellaneous
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id: int = 0
    exp_name: str = "test"
    log_dir: str = './logs'
    seed: int = 42    
    verbose: bool = True

########################################################################################
########################################################################################

if __name__ == "__main__":
    args = Arguments()
    print("=="*60)
    #all_metrics, checkpoint_path = train(args)

    args.n_steps = 10**3 * 1 + 1
    all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=None)
    print("=="*60)
    print("Experiment finished.")