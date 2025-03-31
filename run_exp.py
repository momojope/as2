"""
# Usage
# python run_exp.py --p 31 --operator + --r_train 0.6 --train_batch_size 2**12 --eval_batch_size 2**12 --model gpt --optimizer adamw --lr 1e-3 --momentum 0.9 --weight_decay 1e-1 --n_steps 10**4 * 2 --device cuda --exp_id 0 --exp_name debug --log_dir ../logs --seed 42 --verbose
"""
import torch
from train import train
import argparse

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in {'off', 'false', '0'}:
        return False
    elif s.lower() in {'on', 'true', '1'}:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an experiment for assignment 2.")

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--p",
        type=int,
        default=31,
        help="maximum number of digits in the arithmetic expression (default: %(default)s).",
    )
    data.add_argument(
        "--operator",
        type=str,
        default="+",
        choices=["+", "-", "*", "/"],
        help="arithmetic operator to use (default: %(default)s).",
    )
    data.add_argument(
        "--r_train",
        type=float,
        default=0.5,
        help="ratio of training data (default: %(default)s).",
    )
    data.add_argument(
        "--operation_orders",
        type=int,
        nargs="+",
        choices=[2, 3, [2, 3]],
        default=[2],
        help="list of orders of operations to use (default: %(default)s).",
    )
    data.add_argument(
        "--train_batch_size",
        type=int,
        default=512,
        help="batch size for training (default: %(default)s).",
    )
    data.add_argument(      
        "--eval_batch_size",
        type=int,
        default=2**12,
        help="batch size for evaluation (default: %(default)s).",
    )
    data.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of processes to use for data loading (default: %(default)s).",
    )

    # Model
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model", 
        type=str,
        default="lstm",
        choices=["lstm", "gpt"],
        help="name of the model to run (default: %(default)s).",
    )
    model.add_argument(
        "--num_heads", 
        type=int,
        default=4,
        help="number of heads in the  transformer model (default: %(default)s).",
    )
    model.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="number of layers in the model (default: %(default)s).",
    )
    model.add_argument(
        "--embedding_size",
        type=int,
        default=2**7,
        help="embeddings dimension (default: %(default)s).",
    )
    model.add_argument(
        "--hidden_size",
        type=int,
        default=2**7,
        help="hidden size of the lstm model (default: %(default)s).",
    )
    model.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout rate (default: %(default)s).",
    )
    model.add_argument(
        "--share_embeddings",
        type=bool_flag,
        default=False,
        help="share embeddings between the embedding and the classifier (default: %(default)s).",
    )
    model.add_argument(
        "--bias_classifier",
        type=bool_flag,
        default=True,
        help="use bias in the classifier (default: %(default)s).",
    )

    # Optimization
    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="optimizer name (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate for the optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for the SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=1e-0,
        help="weight decay (default: %(default)s).",
    )

    # Training
    training = parser.add_argument_group("Training")
    training.add_argument(
        "--n_steps",
        type=int,
        default=10**4+1,
        help="number of training steps (default: %(default)s).",
    )
    training.add_argument(
        "--eval_first",
        type=int,
        default=10**2,
        help="Evaluate the model continuously for the first n steps (default: %(default)s).",
    )
    training.add_argument(
        "--eval_period",
        type=int,
        default=10**2,
        help="Evaluate the model every n steps (default: %(default)s).",
    )
    training.add_argument(
        "--print_step",
        type=int,
        default=10**2,
        help="print the training loss every n steps (default: %(default)s).",
    )
    training.add_argument(
        "--save_model_step",
        type=int,
        default=10**3,
        help="save the model every n steps (default: %(default)s).",
    )
    training.add_argument(
        "--save_statistic_step",
        type=int,
        default=10**3,
        help="save the statistics every n steps (default: %(default)s).",
    )

    # Experiment & Miscellaneous
    misc = parser.add_argument_group("Experiment & Miscellaneous")
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to store tensors on (default: %(default)s).",
    )
    misc.add_argument(
        "--exp_id",
        type=int,
        default=0,
        help="experiment id (default: %(default)s).",
    )
    misc.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="experiment name (default: %(default)s).",
    )
    misc.add_argument(
        "--log_dir",
        type=str,
        default="../logs",
        help="directory to save the logs (default: %(default)s).",
    )
    misc.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: %(default)s).",
    )
    misc.add_argument(
        "--verbose", action="store_true", help="print additional information."
    )

    args = parser.parse_args()
    all_metrics, checkpoint_path = train(args)

    print("=="*60)
    print("Experiment finished.")
