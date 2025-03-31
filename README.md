## Clone the repository
```bash
git clone https://github.com/Tikquuss/IFT6135_W25_A2_release
cd IFT6135_W25_A2_release
pip install -r requirements.txt
```

## Train a model
You can use (see [run_exp.py](run_exp.py) for more parameters)
```bash
python run_exp.py --p 31 --operator + --r_train 0.5 --model lstm --optimizer adamw
```
Or (see the class `Arguments` from [train.py](train.py) for more parameters)
```python
import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs

args=Arguments()

# Data
args.p=31
args.operator = "+" # ["+", "-", "*", "/"]
args.r_train = .5
args.operation_orders = 2 # 2, 3 or [2, 3]
args.train_batch_size = 512
args.eval_batch_size = 2**12
args.num_workers = 0

# Model
args.model = 'lstm'  # [lstm, gpt]
args.num_heads = 4
args.num_layers = 2
args.embedding_size = 2**7
args.hidden_size = 2**7
args.dropout = 0.0
args.share_embeddings = False
args.bias_classifier = True

# Optimization
args.optimizer = 'adamw'  # [sgd, momentum, adam, adamw]
args.lr = 1e-3
args.weight_decay = 1e-0

# Training
args.n_steps = 10**4 + 1
args.eval_first = 10**2
args.eval_period = 10**2
args.print_step= 10**2
args.save_model_step = 10**3
args.save_statistic_step = 10**3

# Experiment & Miscellaneous
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.exp_id = 0
args.exp_name = "test"
args.log_dir = '../logs'
args.seed = 42 
args.verbose = True

## Train a single model (one seed)
all_metrics, checkpoint_path = train(args)
## all_metrics contains the training/test loss/accuracies, training steps, etc
plot_loss_accs(all_metrics, multiple_runs=False, log_x=False, log_y=False, fileName=args.exp_name, filePath=None, show=True)

## Train multiple models (multiple seeds)
all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])
plot_loss_accs(all_metrics, multiple_runs=True, log_x=False, log_y=False, fileName=args.exp_name, filePath=None, show=True)
```

## Load checkpoints
```python
from checkpointing import get_all_checkpoints, get_all_checkpoints_per_trials

args.exp_id = 0
args.exp_name = "test"
args.log_dir = '../logs'

## For a single run
checkpoint_path = os.path.join(args.log_dir, str(args.exp_id))
all_models, all_metrics = get_all_checkpoints(checkpoint_path, args.exp_name, just_files=True)

## For a multiple runs
all_checkpoint_paths = []
all_models_per_trials, all_metrics = get_all_checkpoints_per_trials(all_checkpoint_paths, args.exp_name, just_files=True)
```

## Plot informations
```python
from plotter import plot_loss_accs

## For a single run
plot_loss_accs(all_metrics, multiple_runs=False, log_x=False, log_y=False, fileName=args.exp_name, filePath=None, show=True)

## For a multiple run
plot_loss_accs(all_metrics, multiple_runs=True, log_x=False, log_y=False, fileName=args.exp_name, filePath=None, show=True)
```

## Calculate best performances & the steps at which they were achieved (for the first time)

```python
from checkpointing import get_extrema_performance_steps, get_extrema_performance_steps_per_trials

## For a single run
extrema_performances = get_extrema_performance_steps(all_metrics)

## For a multiple run
extrema_performances = get_extrema_performance_steps_per_trials(all_metrics)
```