import torch
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import time


########################################################################################
########################################################################################

def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction='mean'):
    """
    Computes the negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        Logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        Target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence.
    mask : torch.LongTensor of shape (B, S)
        Mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Negative log-likelihood loss computed over the valid RHS tokens.
    accuracy : torch.Tensor
        Accuracy over the batch where a sequence is correct only if all valid RHS tokens are predicted correctly.
    """
    # Get device from input tensors
    device = logits.device

    # Ensure all input tensors are on the same device
    targets = targets.to(device)
    eq_positions = eq_positions.to(device)
    mask = mask.to(device)

    # Create right-hand side mask (ensure it's on the correct device)
    batch_size, seq_length = mask.shape
    rhs_mask = torch.zeros_like(mask, dtype=torch.bool).to(device)

    # Create RHS mask for each sequence
    for i in range(batch_size):
        # RHS starts after '=' token
        rhs_start = eq_positions[i] + 1

        # Find the last valid token in the sequence
        valid_tokens = mask[i].nonzero()
        if valid_tokens.numel() > 0:
            rhs_end = valid_tokens[-1].item() + 1
            rhs_mask[i, rhs_start:rhs_end] = mask[i, rhs_start:rhs_end]

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probabilities of correct tokens
    gathered_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Apply RHS mask to log probabilities (ensure both are on the same device)
    gathered_log_probs = gathered_log_probs.to(device)
    rhs_mask = rhs_mask.to(device)
    masked_log_probs = gathered_log_probs * rhs_mask

    # Compute per-sample loss
    sample_lengths = rhs_mask.sum(dim=-1).float()

    # Avoid division by zero
    sample_lengths = torch.max(sample_lengths, torch.tensor(1.0, device=device))

    sample_losses = -masked_log_probs.sum(dim=-1) / sample_lengths

    # Compute per-sample accuracy
    sample_preds = logits.argmax(dim=-1)

    # Explicit accuracy calculation
    sample_corrects = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for i in range(batch_size):
        # Get RHS indices
        rhs_indices = rhs_mask[i].nonzero().flatten()

        # Check if there are any RHS tokens
        if rhs_indices.numel() > 0:
            # Check if all RHS tokens are correctly predicted
            sample_corrects[i] = torch.all(
                sample_preds[i, rhs_indices] == targets[i, rhs_indices]
            )
        else:
            # If no RHS tokens, consider it incorrect
            sample_corrects[i] = False

    # Apply reduction
    if reduction == 'mean':
        loss = sample_losses.mean()
        accuracy = sample_corrects.float().mean().item()
    elif reduction == 'sum':
        loss = sample_losses.sum()
        accuracy = sample_corrects.float().sum().item()
    else:  # 'none'
        loss = sample_losses
        accuracy = sample_corrects.float()

    return loss, accuracy


# Modified get_loss_and_accuracy function that can separate by operation type
def get_loss_and_accuracy_by_operation(model, dataloader, device, eq_positions_types=[3, 5]):
    model.eval()
    results = {eq_pos: {'loss': 0.0, 'accuracy': 0.0, 'count': 0} for eq_pos in eq_positions_types}

    with torch.no_grad():
        for batch_x, batch_y, eq_positions, mask in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            eq_positions = eq_positions.to(device)
            mask = mask.to(device)

            logits, *_ = model(batch_x)

            # Get per-sample loss and accuracy
            loss, accuracy = get_loss_and_accuracy(logits, batch_y, eq_positions, mask, reduction='none')

            # Group by operation type (eq_positions = 3 for binary, 5 for ternary)
            for eq_pos in eq_positions_types:
                indices = (eq_positions == eq_pos).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    results[eq_pos]['loss'] += loss[indices].sum().item()
                    results[eq_pos]['accuracy'] += accuracy[indices].sum().item()
                    results[eq_pos]['count'] += len(indices)

    # Calculate averages
    for eq_pos in eq_positions_types:
        if results[eq_pos]['count'] > 0:
            results[eq_pos]['loss'] /= results[eq_pos]['count']
            results[eq_pos]['accuracy'] /= results[eq_pos]['count']

    return results


# Modified train function to track loss/accuracy per operation type
def train_with_operation_tracking(model, train_loader, valid_loader, optimizer, scheduler, device, exp_name, checkpoint_path, n_steps, eq_positions_types=[3, 5]):
    os.makedirs(checkpoint_path, exist_ok=True)

    # Metrics to track
    all_metrics = {
        'steps': [],
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'train_binary_loss': [], 'train_binary_acc': [],
        'train_ternary_loss': [], 'train_ternary_acc': [],
        'val_binary_loss': [], 'val_binary_acc': [],
        'val_ternary_loss': [], 'val_ternary_acc': []
    }

    # Evaluate initial model
    train_by_operation = get_loss_and_accuracy_by_operation(model, train_loader, device, eq_positions_types)
    val_by_operation = get_loss_and_accuracy_by_operation(model, valid_loader, device, eq_positions_types)

    # Track initial metrics
    all_metrics['steps'].append(0)
    all_metrics['train_loss'].append(sum(train_by_operation[eq_pos]['loss'] * train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
    sum(train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))
    all_metrics['train_acc'].append(sum(train_by_operation[eq_pos]['accuracy'] * train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
    sum(train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))
    all_metrics['val_loss'].append(sum(val_by_operation[eq_pos]['loss'] * val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
    sum(val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))
    all_metrics['val_acc'].append(sum(val_by_operation[eq_pos]['accuracy'] * val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
    sum(val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))

    # Track by operation type
    all_metrics['train_binary_loss'].append(train_by_operation[3]['loss'])
    all_metrics['train_binary_acc'].append(train_by_operation[3]['accuracy'])
    all_metrics['train_ternary_loss'].append(train_by_operation[5]['loss'])
    all_metrics['train_ternary_acc'].append(train_by_operation[5]['accuracy'])
    all_metrics['val_binary_loss'].append(val_by_operation[3]['loss'])
    all_metrics['val_binary_acc'].append(val_by_operation[3]['accuracy'])
    all_metrics['val_ternary_loss'].append(val_by_operation[5]['loss'])
    all_metrics['val_ternary_acc'].append(val_by_operation[5]['accuracy'])

    # Training loop
    step = 0
    pbar = tqdm(total=n_steps, desc="Training")

    while step < n_steps:
        for batch_x, batch_y, eq_positions, mask in train_loader:
            if step >= n_steps:
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            eq_positions = eq_positions.to(device)
            mask = mask.to(device)

            # Training step
            optimizer.zero_grad()
            model.train()
            logits, *_ = model(batch_x)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate every 100 steps or at the end
            if step % 100 == 0 or step == n_steps - 1:
                model.eval()
                with torch.no_grad():
                    train_by_operation = get_loss_and_accuracy_by_operation(model, train_loader, device, eq_positions_types)
                    val_by_operation = get_loss_and_accuracy_by_operation(model, valid_loader, device, eq_positions_types)

                # Track metrics
                all_metrics['steps'].append(step + 1)
                all_metrics['train_loss'].append(sum(train_by_operation[eq_pos]['loss'] * train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
                sum(train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))
                all_metrics['train_acc'].append(sum(train_by_operation[eq_pos]['accuracy'] * train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
                sum(train_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))
                all_metrics['val_loss'].append(sum(val_by_operation[eq_pos]['loss'] * val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
                sum(val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))
                all_metrics['val_acc'].append(sum(val_by_operation[eq_pos]['accuracy'] * val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types) /
                sum(val_by_operation[eq_pos]['count'] for eq_pos in eq_positions_types))

                # Track by operation type
                all_metrics['train_binary_loss'].append(train_by_operation[3]['loss'])
                all_metrics['train_binary_acc'].append(train_by_operation[3]['accuracy'])
                all_metrics['train_ternary_loss'].append(train_by_operation[5]['loss'])
                all_metrics['train_ternary_acc'].append(train_by_operation[5]['accuracy'])
                all_metrics['val_binary_loss'].append(val_by_operation[3]['loss'])
                all_metrics['val_binary_acc'].append(val_by_operation[3]['accuracy'])
                all_metrics['val_ternary_loss'].append(val_by_operation[5]['loss'])
                all_metrics['val_ternary_acc'].append(val_by_operation[5]['accuracy'])

                # Save metrics
                torch.save(all_metrics, os.path.join(checkpoint_path, f"{exp_name}_metrics.pt"))

                # Save model
                if step % 1000 == 0 or step == n_steps - 1:
                    state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }

########################################################################################
########################################################################################

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    acc = 0
    loss = 0
    n = 0
    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch  # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x)  # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]

    ##########
    # You can add more metrics in the dictionary (e.g., l2 norm of the parameters, etc.)
    ##########

    return {"loss": loss / n, "accuracy": acc / n}


########################################################################################
########################################################################################


def train(
        model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device,
        exp_name: str, checkpoint_path: str,
        n_steps: int, eval_first: int = 0, eval_period: int = 1, print_step: int = 1, save_model_step: int = 1,
        save_statistic_step: int = 1,
        verbose=True,
):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    exp_name (str) : experiment name
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    n_steps (int) : Number of training steps
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    n_steps = total_epochs * len(train_loader)

    if verbose:
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############

    all_metrics = defaultdict(lambda: [])  # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: [])  # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: [])  # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items():
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items():
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0

    ######################
    # Save model
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state,
    f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")

    ##############

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if verbose:
        to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
        to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
        to_print += f" | lr = {current_lr}"
        print(to_print)

    ##############

    cur_step = 1
    tol_step = 0

    for epoch in tqdm(range(1, total_epochs + 1), desc="Training", total=total_epochs):

        # start_time = time.time()

        for i, batch in enumerate(train_loader):
            batch_x, batch_y, eq_positions, mask = batch  # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x)  # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================
            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            # ==========================
            # ==========================

            if cur_step in [1, n_steps] or cur_step % eval_period == 0 or cur_step <= eval_first:
                train_statistics = eval_model(model, train_loader_for_eval, device)
                for k, v in train_statistics.items(): all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device)
                for k, v in test_statistics.items(): all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

            if verbose and (cur_step in [1, n_steps] or cur_step % print_step == 0):
                to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
                to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
                to_print += f" | lr = {current_lr}"
                print(to_print)

            if cur_step in [1, n_steps] or cur_step % save_model_step == 0 or cur_step <= eval_first:
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state,
                f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")

            if cur_step in [1, n_steps] or cur_step % save_statistic_step == 0:
                # to_save = {k:v for k, v in all_metrics.items()}
                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in
                all_metrics.items()}  # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

            cur_step += 1

        # ==========================
        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        # ==========================

        ##############
        # You can implement early stopping here.
        # That is, if the model does not improve for a certain number of steps, you can stop the training.
        ##############

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time for one step : {elapsed_time} seconds")

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state,
               f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")

    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items(): all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items(): all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in
               all_metrics.items()}  # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics
