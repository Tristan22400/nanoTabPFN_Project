import argparse
import copy
import csv
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import h5py
import numpy as np
import schedulefree
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import NanoTabPFNClassifier, NanoTabPFNModel, NanoTabPFNDSAModel

def set_randomness_seed(seed: int):
    """Sets the randomness seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device() -> torch.device:
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class CSVLogger:
    """Simple CSV Logger for tracking training metrics."""
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        if not os.path.isfile(filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        """Logs a row of data to the CSV file."""
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

def get_benchmark_datasets() -> List[Tuple]:
    """Loads and returns benchmark datasets for evaluation."""
    datasets = []
    # Breast Cancer Dataset
    X, y = load_breast_cancer(return_X_y=True)
    datasets.append(train_test_split(X, y, test_size=0.5, random_state=0))
    return datasets

def evaluate_model(classifier: Any, datasets: Optional[List[Tuple]] = None) -> Dict[str, float]:
    """
    Evaluates the classifier on a list of datasets.

    Args:
        classifier: The trained classifier (sklearn-compatible).
        datasets: List of (X_train, X_test, y_train, y_test) tuples.

    Returns:
        Dictionary of averaged scores (ROC AUC, Accuracy, Balanced Accuracy).
    """
    if datasets is None:
        datasets = get_benchmark_datasets()

    scores = {
        "roc_auc": 0.0,
        "acc": 0.0,
        "balanced_acc": 0.0
    }
    
    for X_train, X_test, y_train, y_test in datasets:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1) 
        
        # Handle binary classification for ROC AUC
        if prob.shape[1] == 2:
            prob_score = prob[:, 1]
        else:
            prob_score = prob

        scores["roc_auc"] += float(roc_auc_score(y_test, prob_score, multi_class="ovr"))
        scores["acc"] += float(accuracy_score(y_test, pred))
        scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))

    # Average scores
    num_datasets = len(datasets)
    scores = {k: v / num_datasets for k, v in scores.items()}
    return scores

def train_base_model(
    model: NanoTabPFNModel, 
    prior: DataLoader,
    lr: float = 1e-4, 
    device: torch.device = None, 
    steps_per_eval: int = 10, 
    patience: int = 5,
    eval_func: Optional[Callable] = None,
    logger: Optional[CSVLogger] = None
) -> Tuple[NanoTabPFNModel, List]:
    """
    Trains the base NanoTabPFN model using ScheduleFree optimization.

    Args:
        model: The NanoTabPFN model to train.
        prior: DataLoader providing synthetic prior data.
        lr: Learning rate.
        device: Torch device.
        steps_per_eval: Number of steps between internal evaluations.
        patience: Number of evaluations with no improvement before stopping.
        logger: Optional CSVLogger for tracking metrics.

    Returns:
        Tuple of (trained_model, eval_history).
    """
    if not device:
        device = get_default_device()
    model.to(device)
    
    # ScheduleFree Optimizer
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0.0
    eval_history = []
    
    print(f"Starting Base Training (Early Stopping, Patience={patience})...")

    best_score = 0.0
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())

    try:
        for step, full_data in enumerate(prior):

            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            
            # Prepare Data
            data = (full_data["x"].to(device),
                    full_data["y"][:, :train_test_split_index].to(device))
            targets = full_data["y"].to(device)
            # Target is the rest of the sequence
            targets = targets[:, train_test_split_index:].reshape((-1,)).to(torch.long)

            # --- 2. Training Step ---
            optimizer.zero_grad()
            
            output = model(data, train_test_split_index=train_test_split_index)
            if isinstance(output, tuple): output = output[0]
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()
            
            # Basic NaN Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()} at step {step}. Skipping.")
                continue

            loss.backward()
            
            total_loss = loss.cpu().detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            # --- 3. Internal Evaluation Step ---
            if step % steps_per_eval == steps_per_eval - 1 and eval_func is not None:
                
                # Switch to Eval Mode (Activates Consensus Weights)
                optimizer.eval()
                model.eval()

                classifier = NanoTabPFNClassifier(model, device)
                scores = eval_func(classifier)
                eval_history.append((train_time, scores))
                score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")

                if logger:
                    log_row = {
                        'stage': 'base_train',
                        'step': step,
                        'time': train_time,
                        'loss': total_loss,
                        'syn_acc': 0, # Not calculating syn_acc anymore
                    }
                    log_row.update(scores)
                    logger.log(log_row)

                # --- Early Stopping Check (Maximize ROC AUC) ---
                current_score = scores.get('acc', 0.0)
                if current_score > best_score:
                    best_score = current_score
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    print(f"No improvement in Score. Patience: {patience_counter}/{patience}")
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {step} steps.")
                    model.load_state_dict(best_model_state)
                    break

                # Restore Train Mode
                model.train()
                optimizer.train()

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        pass

    return model, eval_history

def train_indexer_warmup(
    model: NanoTabPFNDSAModel, 
    prior: DataLoader, 
    device: torch.device, 
    lr: float = 1e-4, 
    patience: int = 5,
    logger: Optional[CSVLogger] = None
):
    """
    Warms up the indexer using distillation from the dense attention teacher.

    Args:
        model: The NanoTabPFNDSAModel.
        prior: DataLoader.
        device: Torch device.
        lr: Learning rate.
        patience: Number of steps with no improvement before stopping.
        logger: Optional CSVLogger.
    """
    print(f"Starting Indexer Warmup (Early Stopping, Patience={patience})...")
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    optimizer.train()
    
    train_time = 0.0
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    for step, full_data in enumerate(prior):
            
        step_start = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
        
        optimizer.zero_grad()
        
        # Warmup mode: Student (Indexer) learns from Teacher (Dense Attention)
        _, aux_data_list = model(data, train_test_split_index=train_test_split_index, mode='warmup')
        
        loss = torch.tensor(0.0, device=device)
        for aux in aux_data_list:
            if 'indexer_scores' in aux and 'dense_scores' in aux:
                # --- ROBUST DISTILLATION (KL DIVERGENCE) ---
                # 1. Teacher (Dense): Softmax to get probabilities
                dense_logits = aux['dense_scores'].mean(dim=1) # Average heads
                target_probs = torch.nn.functional.softmax(dense_logits, dim=-1)
                
                # 2. Student (Indexer): LogSoftmax for stability
                indexer_log_probs = torch.nn.functional.log_softmax(aux['indexer_scores'], dim=-1)
                
                # 3. KL Divergence
                loss += torch.nn.functional.kl_div(indexer_log_probs, target_probs, reduction='batchmean')
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Warmup Loss is {loss.item()} at step {step}. Skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_time += time.time() - step_start
        
        if step % 10 == 0:
            print(f"Warmup Step {step} | Time {train_time:.1f}s | Loss {loss.item():.4f}")
            if logger:
                logger.log({
                    'stage': 'indexer_warmup',
                    'step': step,
                    'time': train_time,
                    'loss': loss.item(),
                    'syn_acc': 0, 'acc': 0, 'roc_auc': 0, 'balanced_acc': 0 
                })
            
            # --- Early Stopping Check (using training loss as proxy for warmup) ---
            # Note: Ideally we'd use a validation set, but for warmup, low loss is the goal.
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Warmup Early stopping triggered after {step} steps.")
                model.load_state_dict(best_model_state)
                break

def train_sparse_finetune(
    model: NanoTabPFNDSAModel, 
    prior: DataLoader, 
    device: torch.device, 
    lr: float = 1e-4, 
    steps_per_eval: int = 30, 
    patience: int = 5,
    eval_func: Optional[Callable] = None,
    logger: Optional[CSVLogger] = None
) -> Tuple[NanoTabPFNDSAModel, List]:
    """
    Finetunes the model using sparse attention.

    Args:
        model: The NanoTabPFNDSAModel.
        prior: DataLoader.
        device: Torch device.
        lr: Learning rate.
        steps_per_eval: Steps between evaluations.
        patience: Number of evaluations with no improvement before stopping.
        logger: Optional CSVLogger.

    Returns:
        Tuple of (trained_model, eval_history).
    """
    print(f"Starting Sparse Finetune (Early Stopping, Patience={patience})...")
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    optimizer.train()
    
    train_time = 0.0
    eval_history = []

    best_score = 0.0
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    for step, full_data in enumerate(prior):
            
        step_start = time.time()
        train_test_split_index = full_data["train_test_split_index"]
        
        data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
        targets = full_data["y"].to(device)
        targets = targets[:, train_test_split_index:].reshape((-1,)).to(torch.long)
        
        optimizer.zero_grad() 

        # Sparse Train Mode (Joint Training)
        output, aux_data_list = model(data, train_test_split_index=train_test_split_index, mode='joint_train')
        if isinstance(output, tuple): output = output[0]
        output = output.view(-1, output.shape[-1])
        
        loss_main = criterion(output, targets).mean()
        
        # Auxiliary Loss (Indexer Distillation)
        loss_aux = torch.tensor(0.0, device=device)
        for aux in aux_data_list:
            if 'indexer_scores' in aux and 'dense_scores' in aux:
                # Teacher (Dense) - Detached
                dense_logits = aux['dense_scores'].mean(dim=1).detach()
                target_probs = torch.nn.functional.softmax(dense_logits, dim=-1)
                
                # Student (Indexer)
                indexer_log_probs = torch.nn.functional.log_softmax(aux['indexer_scores'], dim=-1)
                
                loss_aux += torch.nn.functional.kl_div(indexer_log_probs, target_probs, reduction='batchmean')
        
        loss = loss_main + loss_aux

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Loss is {loss.item()} at step {step}. Skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_time += time.time() - step_start  
        
        # Evaluation
        if step % steps_per_eval == steps_per_eval - 1 and eval_func is not None:
            
            optimizer.eval() 
            model.eval() 
            
            classifier = NanoTabPFNClassifier(model, device)
            scores = eval_func(classifier)
            eval_history.append((train_time, scores))
            score_str = " | ".join([f"{k} {v:7.4f}" for k,v in scores.items()])
            print(f"time {train_time:7.1f}s | loss {loss.item():7.4f} | {score_str}")

            if logger:
                log_row = {
                    'stage': 'sparse_finetune',
                    'step': step,
                    'time': train_time,
                    'loss': loss.item(),
                    'syn_acc': 0,
                }
                log_row.update(scores)
                logger.log(log_row)
            
            # --- Early Stopping Check ---
            current_score = scores.get('acc', 0.0)
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                print(f"No improvement in Score. Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {step} steps.")
                model.load_state_dict(best_model_state)
                break
            
            model.train()
            optimizer.train()
    
    # Final consistency set
    optimizer.eval() 
    model.eval()
    
    return model, []


def train_coordinate_block_descent(
    model: NanoTabPFNDSAModel, 
    prior: DataLoader, 
    device: torch.device, 
    lr_indexer: float = 1e-4, 
    lr_tfm: float = 1e-4,
    n_indexer_steps: int = 10,
    n_tfm_steps: int = 10,
    logger: Optional[CSVLogger] = None
) -> Tuple[NanoTabPFNDSAModel, List]:
    """
    Implements Coordinate Block Descent (CBD) training for NanoTabPFNDSAModel.
    
    Theory:
    Optimization of Sparse Attention models involves two distinct sets of parameters with different dynamics:
    1. Indexer Parameters (theta_I): Select which tokens to attend to. Hard to train end-to-end due to discrete selection.
    2. TFM Parameters (theta_M): The main model weights (Encoders, Attention, MLPs).
    
    CBD alternates between two blocks:
    - Block 1 (Indexer Step): Fix theta_M, Minimize L_distill(theta_I; theta_M_fixed).
      We use the dense attention scores (from theta_M) as a teacher to train the Indexer.
    - Block 2 (TFM Step): Fix theta_I, Minimize L_task(theta_M; theta_I_fixed).
      We use the fixed Indexer to select indices, and train the model on the main task (CrossEntropy).
      
    Args:
        model: The NanoTabPFNDSAModel.
        prior: DataLoader.
        device: Torch device.
        lr_indexer: Learning rate for Indexer.
        lr_tfm: Learning rate for TFM.
        n_indexer_steps: Number of steps per Indexer block.
        n_tfm_steps: Number of steps per TFM block.
        logger: Optional CSVLogger.
    """
    print(f"\n{'='*20}\nStarting Coordinate Block Descent Training\n{'='*20}")
    print(f"Configuration: Indexer Steps={n_indexer_steps} (LR={lr_indexer}), TFM Steps={n_tfm_steps} (LR={lr_tfm})")
    
    # 1. Parameter Splitting
    indexer_params = []
    tfm_params = []
    for name, param in model.named_parameters():
        if 'indexer' in name:
            indexer_params.append(param)
        else:
            tfm_params.append(param)
            
    # 2. Optimizers
    opt_indexer = schedulefree.AdamWScheduleFree(indexer_params, lr=lr_indexer, weight_decay=0.0)
    opt_tfm = schedulefree.AdamWScheduleFree(tfm_params, lr=lr_tfm, weight_decay=0.0)
    
    criterion = nn.CrossEntropyLoss()
    
    train_time = 0.0
    
    # Ensure model is on device
    model.to(device)
    
    cycle_len = n_indexer_steps + n_tfm_steps
    
    try:
        for step, full_data in enumerate(prior):
            step_start = time.time()
            
            cycle_pos = step % cycle_len
            
            # --- BLOCK SELECTION ---
            if cycle_pos < n_indexer_steps:
                # --- BLOCK 1: INDEXER OPTIMIZATION ---
                mode = 'indexer'
                opt_indexer.train()
                opt_tfm.eval()
                model.train()
                
                train_test_split_index = full_data["train_test_split_index"]
                data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
                
                opt_indexer.zero_grad()
                
                _, aux_data_list = model(data, train_test_split_index=train_test_split_index, mode='warmup')
                
                loss = torch.tensor(0.0, device=device)
                for aux in aux_data_list:
                    if 'indexer_scores' in aux and 'dense_scores' in aux:
                        dense_logits = aux['dense_scores'].mean(dim=1).detach()
                        target_probs = torch.nn.functional.softmax(dense_logits, dim=-1)
                        indexer_log_probs = torch.nn.functional.log_softmax(aux['indexer_scores'], dim=-1)
                        loss += torch.nn.functional.kl_div(indexer_log_probs, target_probs, reduction='batchmean')
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(indexer_params, 1.0)
                    opt_indexer.step()
                    
                log_stage = 'cbd_indexer'

            else:
                # --- BLOCK 2: TFM OPTIMIZATION ---
                mode = 'tfm'
                opt_indexer.eval()
                opt_tfm.train()
                model.train()
                
                train_test_split_index = full_data["train_test_split_index"]
                data = (full_data["x"].to(device), full_data["y"][:, :train_test_split_index].to(device))
                targets = full_data["y"].to(device)
                targets = targets[:, train_test_split_index:].reshape((-1,)).to(torch.long)
                
                opt_tfm.zero_grad()
                
                output, _ = model(data, train_test_split_index=train_test_split_index, mode='sparse_train')
                if isinstance(output, tuple): output = output[0]
                output = output.view(-1, output.shape[-1])
                
                loss = criterion(output, targets).mean()
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tfm_params, 1.0)
                    opt_tfm.step()
                    
                log_stage = 'cbd_tfm'

            train_time += time.time() - step_start
            
            if step % 10 == 0:
                print(f"Step {step:4d} | Block: {mode.upper():7s} | Time: {train_time:6.1f}s | Loss: {loss.item():.4f}")
                if logger:
                    logger.log({'stage': log_stage, 'step': step, 'time': train_time, 'loss': loss.item()})

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        
    # Final consistency
    opt_indexer.eval()
    opt_tfm.eval()
    model.eval()
    
    return model, []

class PriorDumpDataLoader(DataLoader):
    """
    DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename: str, num_steps: int, batch_size: int, device: torch.device = None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0
        if device is None:
            self.device = get_default_device()
        
        # Open file to read metadata
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoTabPFN models.")
    parser.add_argument("--model_type", type=str, choices=["base", "dsa", "both", "cbd"], default="dsa", help="Type of model to train")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    args = parser.parse_args()

    device = get_default_device()
    
    models_to_run = []
    if args.model_type == "both":
        models_to_run = ["base", "dsa"]
    else:
        models_to_run = [args.model_type]

    for m_type in models_to_run:
        print(f"\n{'='*20}\nTraining {m_type} model for {args.num_steps} steps\n{'='*20}")
        
        # 1. Re-initialize data loader for each run to ensure fairness
        prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=args.num_steps, batch_size=32, device=device)
        
        # 2. Setup Logger
        # We added 'syn_acc' (Synthetic Accuracy) to the logs as it's our primary metric now
        logger = CSVLogger(f"training_log_{m_type}.csv", fieldnames=['stage', 'step', 'time', 'loss', 'syn_acc', 'acc', 'roc_auc', 'balanced_acc'])

        if m_type == "base":
            # --- Base Model Configuration ---
            model = NanoTabPFNModel(
                embedding_size=96,
                num_attention_heads=4,
                mlp_hidden_size=192,
                num_layers=3,
                num_outputs=2
            )
            
            # The 'train' function now handles max_time and internal synthetic eval
            # We use a higher LR (4e-3) for base training from scratch vs finetuning (1e-5)
            model, history = train_base_model(
                model, 
                prior, 
                lr=4e-3, 
                steps_per_eval=25, 
                patience=5,
                eval_func=evaluate_model,
                logger=logger
            )
            
        elif m_type == "dsa":
            # --- DSA Model Configuration ---
            model = NanoTabPFNDSAModel(
                embedding_size=96,
                num_attention_heads=4,
                mlp_hidden_size=192,
                num_layers=3,
                num_outputs=2,
                top_k=64,
                use_dsa=True
            ).to(device)
            
            # Split steps: 10% Warmup, 90% Finetuning
            warmup_steps = int(0.1 * args.num_steps)
            finetune_steps = args.num_steps - warmup_steps
            
            # Re-init prior for warmup
            prior_warmup = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=warmup_steps, batch_size=32, device=device)
            
            # Stage A: Warmup (Distillation with KL Divergence)
            train_indexer_warmup(
                model, 
                prior_warmup, 
                device, 
                patience=5,
                logger=logger
            )
            
            # Re-init prior for finetune
            prior_finetune = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=finetune_steps, batch_size=32, device=device)
            
            # Stage B: Finetune (Sparse Training with Internal Eval)
            model, history = train_sparse_finetune(
                model, 
                prior_finetune, 
                device, 
                steps_per_eval=25, 
                lr=1e-5,
                patience=5,
                eval_func=evaluate_model,
                logger=logger
            )
            
            
        elif m_type == "cbd":
            # --- Coordinate Block Descent ---
            model = NanoTabPFNDSAModel(
                embedding_size=96,
                num_attention_heads=4,
                mlp_hidden_size=192,
                num_layers=3,
                num_outputs=2,
                top_k=64,
                use_dsa=True
            ).to(device)
            
            model, history = train_coordinate_block_descent(
                model,
                prior,
                device,
                lr_indexer=1e-4,
                lr_tfm=1e-4,
                n_indexer_steps=50, # Train indexer for 50 steps
                n_tfm_steps=50,     # Then train TFM for 50 steps
                logger=logger
            )

        print(f"Final evaluation for {m_type}:")
        # Ensure we switch to eval mode one last time before final external check
        model.eval() 
        print(evaluate_model(NanoTabPFNClassifier(model, device)))