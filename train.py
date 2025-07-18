import argparse
import os

import torch
import numpy as np
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)

# from model_nested import NerFilteredSemiCRF
from pair2rel import Pair2Rel
from pair2rel.modules.run_evaluation import sample_train_data
from pair2rel.model import load_config_as_namespace
from datetime import datetime
import json
import logging
import random
import shutil
import wandb
from functools import partial
from sklearn.model_selection import train_test_split
import time
import gc
import sys
sys.path.append('data/re-docred')
from run_evaluation import run_evaluation
from redocred_experiment_params import REDOCRED_EXP_SWEEP_CONFIG


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

'''

python train.py --config configs/config_conll2004.yaml --wandb_sweep

python train.py --config configs/config_conll2004.yaml

CUDA_VISIBLE_DEVICES="0" python train.py --config configs/config_conll2004.yaml --wandb_sweep --sweep_method grid --experiment


'''

# If doing hyperparameter sweeping, define sweep config here
HP_SWEEP_CONFIG = {
    "metric": {"goal": "maximize", "name": "eval_f1_macro"},
    "parameters": {
        "scorer": {"values": "dot_norm"},
        # "num_train_rel_types": {"values": [15, 20, 25, 30, 35, 40]},
        # "num_unseen_rel_types": {"values": [15]},
        # "random_drop": {"values": [True, False]},
        "lr_encoder": {"values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]},
        "lr_others": {"values": [1e-4, 5e-4, 1e-3, 5e-3]},
        'num_layers_freeze': {"values": [None, 2, 4, 7, 10]},
        "refine_prompt": {"values": [True, False]},
        "refine_relation": {"values": [True, False]},
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "loss_func": {"values": ["binary_cross_entropy_loss", "focal_loss"]},
        "alpha": {"values": [0.3, 0.5, 0.75]},  # focal loss only
        "gamma": {"values": [1, 3, 5]},         # focal loss only
        # "model_name": {"values": ["microsoft/deberta-v3-large", "microsoft/deberta-v3-small"]},
    },
}

EXP_SWEEP_CONFIG = {
    "metric": {"goal": "maximize", "name": "eval_f1_macro"},
    "parameters": {
        'seed': {"values": [11222333]}, 
        # "refine_prompt": {"values": [False, True]},
        # "refine_relation": {"values": [False, True]},
        # "span_marker_mode": {"values": ["markerv1", "markerv2"]},
        # "add_entity_markers": {"values": [False, True]},
        # "label_embed_strategy": {"values": ["label"]},
        # "random_drop": {"values": [False]},
        "num_unseen_rel_types": {"values": [ 10 ]},
        # "subtoken_pooling": {"values": ["mean", "first_last"]},   # "mean", "first_last", "first", "last"  # https://flairnlp.github.io/docs/tutorial-embeddings/transformer-embeddings#pooling-operation
        # "prev_path": {"values": [ ]},  # 
    },
}


def create_parser():
    parser = argparse.ArgumentParser(description="Zero-shot Relation Extraction")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the log directory')
    parser.add_argument("--wandb_log", action="store_true", help="Activate wandb logging")
    parser.add_argument("--wandb_sweep", action="store_true", help="Activate wandb hyperparameter sweep")
    parser.add_argument("--sweep_id", type=str, default=None, help="WandB Sweep ID")
    parser.add_argument("--sweep_method", type=str, default="grid", help="Sweep method (grid, random, bayes)")
    parser.add_argument("--skip_splitting", action="store_true", help="Skip dataset splitting into train and eval sets")
    parser.add_argument("--experiment", action="store_true", help="Run an experiment")
    return parser

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()

def get_unique_relations(data):
    unique_rel_types = []
    for item in data:
        for r in item['relations']:
            unique_rel_types.append(r["relation_text"])
    unique_rel_types = list(set(unique_rel_types))
    return unique_rel_types

from collections import defaultdict

def split_data_by_relation_type(data, num_unseen_rel_types, seed=None):
    """
    Attempts to split a dataset into training and testing sets based on relation types,
    aiming to have a specified number of unique relation types exclusively in the test set
    to simulate a zero-shot learning scenario. The function shuffles and splits the relation
    types, allocating the first chunk as unseen relation types for testing and the rest for training.
    
    It iteratively adjusts the number of unseen relation types if the initial split does not achieve
    the desired number of unique test relation types, retrying with an incremented number until it succeeds
    or the number reaches twice the original request, resetting as needed.

    Notes:
        - This function relies heavily on the assumption that sufficient relation diversity exists
          to meet the zero-shot criteria. If not, the test set may not end up with the intended
          number of unique unseen relation types.
        - The function can potentially skip a significant number of items that contain both train and
          test relation types, leading to data wastage.
        - The iterative process to adjust unseen relation types may lead to computational inefficiency,
          especially for large datasets with diverse relations.
    """

    logger.info("Running efficient dataset splitting...")
    start = time.time()
    
    # Step 1: Map relation types to data items
    rel_to_items = defaultdict(list)
    for item in data:
        for rel in item["relations"]:
            rel_to_items[rel["relation_text"]].append(item)

    all_relations = list(rel_to_items.keys())
    random.seed(seed)
    random.shuffle(all_relations)

    # Step 2: Select unseen relation types
    unseen_rel_types = set(all_relations[:num_unseen_rel_types])
    seen_rel_types = set(all_relations[num_unseen_rel_types:])

    train_data, test_data, skipped = [], [], []

    for item in data:
        item_rel_types = {r["relation_text"] for r in item["relations"]}

        if item_rel_types.issubset(unseen_rel_types):
            test_data.append(item)
        elif item_rel_types.issubset(seen_rel_types):
            train_data.append(item)
        else:
            skipped.append(item)  # Mixed relation types

    logger.info(f"Split done with seed={seed} in {time.time() - start:.2f}s")
    logger.info(f"Train size: {len(train_data)} | Test size: {len(test_data)} | Skipped: {len(skipped)}")

    return train_data, test_data

    
def dirty_split_data_by_relation_type(data, num_unseen_rel_types, max_test_size):
    '''
    This function does not care if the interesection of train and test relation types is empty.
    Used for custom datasets to avoid having a large number of eval classes (causes OOM), 
    and I do not mind if the eval set has some train classes.
    '''
    logger.info("Dirty splitting data...")

    unique_relations = get_unique_relations(data)
    correct_num_unseen_relations_achieved = False
    original_num_unseen_rel_types = num_unseen_rel_types


    
    seed = 400
    random.seed(seed)
    random.shuffle(unique_relations)
    test_relation_types = set(unique_relations[ : num_unseen_rel_types ])
    
    train_data = []
    test_data = []

    # Splitting data based on relation types
    for item in data:
        relation_types = {r["relation_text"] for r in item['relations']}
        if len(test_data) < max_test_size and any([rel in test_relation_types for rel in relation_types]):
            test_data.append(item)
        else:
            train_data.append(item)

    # if we have the right number of eval relations, break
    if len(get_unique_relations(test_data)) == original_num_unseen_rel_types or len(test_data) >= max_test_size: 
        correct_num_unseen_relations_achieved = True
    else:
        # bump the number of unseen relations by 1 to cast a wider net
        # if the bump gets too big, reset it
        num_unseen_rel_types = num_unseen_rel_types + 1 if (num_unseen_rel_types <  original_num_unseen_rel_types*2) else num_unseen_rel_types


    return train_data, test_data


def freeze_n_layers(model, N):
    """
    Freezes or unfreezes the first n layers of the model.
    See DeBERTa model specs here: https://github.com/microsoft/DeBERTa?tab=readme-ov-file#pre-trained-models

    Args:
        model: Assumes model has a DeBERTa model under `model.token_rep_layer`
        n (int): Number of layers to freeze/unfreeze.
        freeze (bool): If True, freeze the layers; if False, unfreeze them.
    """
    # Ensure N is within the valid range
    total_layers = len(model.token_rep_layer.bert_layer.model.encoder.layer)
    if N < 0 or N > total_layers:
        raise ValueError(f"N must be between 0 and total layers ({total_layers}), got {N}")

    # Iterate over the first n layers
    for layer in model.token_rep_layer.bert_layer.model.encoder.layer[:N]:
        for param in layer.parameters():
            param.requires_grad = False

    logger.info(f"Freezing the first {N} layers of the model")
    return model


class EarlyStoppingException(Exception):
    pass


class EarlyStopping:
    def __init__(self, patience, delta, max_saves):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            verbose (bool): If True, prints a message for each validation metric improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            max_saves (int): Maximum number of models to save.
        """
        self.patience = patience
        self.delta = delta
        self.max_saves = max_saves

        self.saved_models = []
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.inf

    def __call__(self, metric, model, save_path) -> None:
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"Validation metric did not improve by delta ({self.delta}): ({self.best_score:.6f} --> {score:.6f}).")
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f'Early stopping at step {self.counter}')
                raise EarlyStoppingException
        else:
            logger.info(f'Validation metric improved!! ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
            self.best_score = score
            self.save_checkpoint(score, model, save_path)
            self.counter = 0

    def save_checkpoint(self, score, model, save_path) -> None:
        '''Saves model when validation metric improves.'''
        self.best_metric = score

        model.save_pretrained(save_path)
        logger.info(f'Model saved at {save_path}')
        self.saved_models.append((save_path, score))

        if len(self.saved_models) > self.max_saves:
            self.saved_models.sort(key=lambda x: (x[1], x[0]), reverse=True) # Sort models by score, then by path
            lowest_f1_model = self.saved_models.pop()                 # Remove the model with the lowest score
            shutil.rmtree(lowest_f1_model[0])
            logger.info(f"Removed model with score at {lowest_f1_model[0]}")
        return

# train function
def train(model, optimizer, train_data, config, train_rel_types, eval_rel_types, eval_data=None, 
          num_steps=1000, eval_every=100, top_k=1, log_dir=None,
          wandb_log=False, wandb_sweep=False, warmup_ratio=0.1, train_batch_size=8, device='cuda', use_amp=True):

    # EarlyStopping
    max_saves = config.max_saves if hasattr(config, 'max_saves') else 3
    patience = config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else None
    patience = patience if patience is not None else 100
    delta = config.early_stopping_delta if hasattr(config, 'early_stopping_delta') else 0.0
    delta = delta if delta is not None else 0.0
    early_stopping = EarlyStopping(patience=patience, delta=delta, max_saves=max_saves)


    if wandb_log:
        # Start a W&B Run with wandb.init
        wandb.login()
        run = wandb.init()
    else:
        run = None
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()

    # freeze params if requested
    if hasattr(config, 'num_layers_freeze') and config.num_layers_freeze is not None:
        model = freeze_n_layers(model, N=config.num_layers_freeze)

    # initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=train_batch_size, shuffle=False, train_relation_types=train_rel_types)

    pbar = tqdm(range(num_steps))

    if warmup_ratio < 1:
        num_warmup_steps = int(num_steps * warmup_ratio)
    else:
        num_warmup_steps = int(warmup_ratio)

    if config.scheduler == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif config.scheduler == "cosine_with_hard_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps,
            num_cycles=3
        )
    else:
        raise ValueError(f"Invalid scheduler: {config.scheduler}")

    iter_train_loader = iter(train_loader)

    prev_model_path = None

    accumulated_steps = 0 
    start = time.time()
    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}


        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            try:
                out = model(x)  # Forward pass
                loss, coref_loss, rel_loss = out['total_loss'], out.get('coref_loss', None), out.get('rel_loss', None)
            except Exception as e:
                logger.error(f"Error in step {step}: {e}")
                logger.error(f"Num tokens: {[len(x['tokens'][i]) for i in range(len(x['tokens']))]}")
                logger.error(f"Num relations: {[x['rel_label'][i].shape[0] for i in range(len(x['rel_label']))]}")
                logger.error(f"Num spans: {[x['span_idx'][i].shape[0] for i in range(len(x['span_idx']))]}")
                logger.error(f"Num candidate classes: {[len(x['classes_to_id'][i]) for i in range(len(x['classes_to_id']))]}")
                continue
        

        # check if loss is nan
        if torch.isnan(loss):
            logger.warning(f"Loss is NaN at step {step}")
            continue

        if config.gradient_accumulation is not None:
            loss = loss / config.gradient_accumulation  # Normalize the loss to account for the accumulation

        try:
            scaler.scale(loss).backward()  # Compute gradients
        except Exception as e:
            logger.error(f"Backprop Loss Error in step {step}: {e}")
            logger.error(f"Num tokens: {[len(x['tokens'][i]) for i in range(len(x['tokens']))]}")
            logger.error(f"Num relations: {[x['rel_label'][i].shape[0] for i in range(len(x['rel_label']))]}")
            logger.error(f"Num spans: {[x['span_idx'][i].shape[0] for i in range(len(x['span_idx']))]}")
            logger.error(f"Num candidate classes: {[len(x['classes_to_id'][i]) for i in range(len(x['classes_to_id']))]}")
            continue

        num_tokens = [len(x['tokens'][i]) for i in range(len(x['tokens']))]
        candidate_classes = [x['classes_to_id'][i] for i in range(len(x['classes_to_id']))]
        status = f"Step {step} | loss: {loss.item()}"
        if coref_loss is not None:
            status += f" | coref_loss: {coref_loss.item()} | rel_loss: {rel_loss.item()}"
        status += f" | x['rel_label']: {x['rel_label'].shape} | x['span_idx']: {x['span_idx'].shape} | x['tokens']: {num_tokens} | num candidate_classes: {[len(x['classes_to_id'][i]) for i in range(len(x['classes_to_id']))]}"
        logger.info(status)

        accumulated_steps += 1
        if config.gradient_accumulation is None or (accumulated_steps % config.gradient_accumulation == 0):
            # optimizer.step()        # Update parameters
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()                        # Update learning rate schedule
            optimizer.zero_grad(set_to_none=True)   # Clear gradients after update (set_to_none=True here can modestly improve performance)
            accumulated_steps = 0                   # Reset accumulation counter


        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"


        if hasattr(config, 'save_at') and (step+1) in config.save_at:
            logger.info(f"Saving model at step {step+1}")
            current_path = os.path.join(log_dir, f'saved_at/model_{step + 1}')

            model.save_pretrained(current_path)

        if run is not None:
            run.log({
                "loss": loss.item(), 
                "num_relations": x['rel_label'].shape[1], 
                "num_tokens": max(num_tokens)
            })

        elif wandb_sweep:
            wandb.log(
                    {
                    "epoch": step // len(train_loader),
                    "train_loss": loss.item(),
                }
            )

        if (step + 1) % eval_every == 0:
            end = time.time()
            logger.info(f"Time taken for {eval_every} steps: {round(end - start)} seconds")
            start = time.time() # reset timer

            model.eval()

            current_path = os.path.join(log_dir, f'model_{step + 1}')

            # if there's no eval data, save the model and remove the previous one
            if eval_data is None:
                if prev_model_path:
                    shutil.rmtree(prev_model_path)

                model.save_pretrained(current_path)
                logger.info(f"Model saved at {current_path}")
                prev_model_path = current_path

            
            elif eval_data is not None:
                with torch.no_grad():

                    wandb_payload = {}

                    # (Re-)DocRED-specific testing
                    if config.dataset_name.lower() == 'redocred':
                        logger.info("Running testing...")
                        test_metrics = run_evaluation(
                            ckpt_dir=log_dir, use_gold_coref=True, 
                            use_auxiliary_coref=False, model=model)
                        test_log_string = "Step={step} | "
                        for k, v in test_metrics.items():
                            test_log_string += f"{k}: {v} | "
                        logger.info(test_log_string)
                        wandb_payload.update(test_metrics)
                    ######


                    logger.info('Evaluating...')
                    logger.info(f'Taking top k = {top_k} predictions for each relation...')

                    results, metric_dict = model.evaluate(
                        eval_data, 
                        flat_ner=True, 
                        threshold=config.eval_threshold, 
                        batch_size=config.eval_batch_size,
                        relation_types=eval_rel_types if config.fixed_relation_types else [],
                        top_k=top_k,
                        dataset_name=config.dataset_name
                    )
                    
                    model.eval()
                    eval_loader = model.create_dataloader(eval_data, batch_size=config.eval_batch_size, shuffle=False, train_relation_types=eval_rel_types if config.fixed_relation_types else [])
                    total_loss = 0
                    num_batches = 0

                    with torch.no_grad():
                        for batch in eval_loader:
                            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                            out = model(batch)
                            total_loss += out['total_loss'].item()
                            num_batches += 1

                    eval_loss = total_loss / max(1, num_batches)
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")
                    wandb_payload["eval_loss"] = eval_loss  # ✅ add to wandb

                    micro_f1, micro_precision, micro_recall = metric_dict['micro_f1'], metric_dict['micro_precision'], metric_dict['micro_recall']
                    macro_f1, macro_precision, macro_recall = metric_dict['macro_f1'], metric_dict['macro_precision'], metric_dict['macro_recall']
                    logger.info(f"Best threshold for eval: {metric_dict['best_threshold']}")

                    wandb_payload.update({
                                "epoch": step // len(train_loader),
                                "eval_f1_micro": micro_f1,
                                "eval_f1_macro": macro_f1,
                                "eval_precision_micro": micro_precision,
                                "eval_precision_macro": macro_precision,
                                "eval_recall_micro": micro_recall,
                                "eval_recall_macro": macro_recall,
                    })

                    if wandb_sweep:
                        wandb.log(wandb_payload)
                    elif run is not None:
                        run.log({"eval_f1_micro": micro_f1, "eval_f1_macro": macro_f1})

                    logger.info(f"Step={step}\n{results}")  

                    early_stopping(metric=metric_dict[model.threshold_search_metric], model=model, save_path=current_path)

            # resume training
            model.train()
            if hasattr(config, 'num_layers_freeze') and config.num_layers_freeze is not None:
                model = freeze_n_layers(model, N=config.num_layers_freeze)
                
            flush_memory()

        pbar.set_description(description)


def main(args):

    # load config
    config = load_config_as_namespace(args.config)
    print(f"Config: {config.dataset_name}")
    config.log_dir = args.log_dir
    seed  = getattr(config, 'seed', None)

    # set up logging
    if config.log_dir is None:
        current_time = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        config.log_dir = f'logs/{config.dataset_name}/{config.dataset_name}-{current_time}'
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    log_file = "train.log"
    log_file_path = os.path.join(config.log_dir, log_file)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Relation extraction training started")


    if args.wandb_sweep:
        run = wandb.init()
        # overwrite config values with sweep values 
        for attribute, v in wandb.config.items():
            logger.info(f"Overwriting {attribute} with {wandb.config[attribute]}")
            setattr(config, attribute, wandb.config[attribute])


    # Prep data
    logger.info(f"Loading data from {config.train_data}...")
    if isinstance(config.train_data, str):
        config.train_data = [config.train_data]

    train_data = []
    
    for train_subset in config.train_data:
        if train_subset.endswith('.jsonl'):
            with open(train_subset, 'r') as f:
                train_subset = [json.loads(line) for line in f]

        elif train_subset.endswith('.json'):
            with open(train_subset, 'r') as f:
                train_subset = json.load(f)
        else:
            raise ValueError(f"Invalid data format: {config.train_data}")
        train_data.extend(train_subset)
    data = train_data
    
    


    if hasattr(config, 'eval_data'):

        if isinstance(config.eval_data, str):
            config.eval_data = [config.eval_data]

        eval_data = []
        for eval_subset in config.eval_data:
            if eval_subset.endswith('.jsonl'):
                with open(eval_subset, 'r') as f:
                    eval_subset = [json.loads(line) for line in f]
            elif eval_subset.endswith('.json'):
                with open(eval_subset, 'r') as f:
                    eval_subset = json.load(f)
            else:
                raise ValueError(f"Invalid data format: {config.eval_data}. Must be .jsonl or .json")
            eval_data.extend(eval_subset)

    else:
        eval_data = None

    # print(len(eval_data))
    # train / eval split

    if eval_data is None:
        if args.skip_splitting:
            print("Skipping dataset splitting. Randomly splitting data into train and eval sets.")
            data = sorted(data, key=lambda x: len(x['relations']))

        elif config.num_unseen_rel_types is not None:

            if config.dataset_name in ['zero_rel_wiki_zsl', 'zero_rel']:
                file_name = 'data/wiki_zsl_all.jsonl'
                config.eval_data = file_name
                with open(file_name, 'r') as f:
                    logger.info(f"Generating eval split from {file_name}...")
                    eval_data = [json.loads(line) for line in f]
                _, eval_data = split_data_by_relation_type(eval_data, config.num_unseen_rel_types, seed=seed)
                data = sorted(data, key=lambda x: len(x['relations']))
                train_data = data
            elif config.dataset_name == 'zero_rel_few_rel':
                file_name = 'data/few_rel_all.jsonl'
                config.eval_data = file_name
                with open(file_name, 'r') as f:
                    logger.info(f"Generating eval split from {file_name}...")
                    eval_data = [json.loads(line) for line in f]
                _, eval_data = split_data_by_relation_type(eval_data, config.num_unseen_rel_types, seed=seed)
                data = sorted(data, key=lambda x: len(x['relations']))
                train_data = data
            else:
                train_data, eval_data = split_data_by_relation_type(data, config.num_unseen_rel_types, seed=seed)
        else:
            raise ValueError("No eval data provided and config.num_unseen_rel_types is None")
    else:
        eval_data = eval_data
        train_data = data

    
    # Load synthetic data
    if hasattr(config, 'synthetic_data') and config.synthetic_data is not None:
        logger.info(f"Loading synthetic data from {config.synthetic_data}...")
        if isinstance(config.synthetic_data, str):
            config.synthetic_data = [config.synthetic_data]

        synthetic_data = []
        for synthetic_subset in config.synthetic_data:
            if synthetic_subset.endswith('.jsonl'):
                with open(synthetic_subset, 'r') as f:
                    synthetic_subset = [json.loads(line) for line in f]

            elif synthetic_subset.endswith('.json'):
                with open(synthetic_subset, 'r') as f:
                    synthetic_subset = json.load(f)
            else:
                raise ValueError(f"Invalid data format: {config.train_data}")
            synthetic_data.extend(synthetic_subset)

        train_data = train_data + synthetic_data


    train_rel_types = get_unique_relations(train_data)
    eval_rel_types = get_unique_relations(eval_data) if eval_data is not None else None
    if train_rel_types==[]:
        train_rel_types=eval_rel_types
    
    logger.info(f"Num Train relation types: {len(train_rel_types)}")
    logger.info(f"Number of train samples: {len(train_data)}")
    if eval_data is not None:
        logger.info(f"Intersection: {set(train_rel_types) & set(eval_rel_types)}")
        logger.info(f"Num Eval relation types: {len(eval_rel_types)}")
        logger.info(f"Number of eval samples: {len(eval_data)}")


    # Load model

    if config.prev_path != "none":
        model = Pair2Rel.from_pretrained(config.prev_path)
        model.config = config
        model.base_config = config
    else:
        model = Pair2Rel(config)


    # Get number of parameters (trainable and total)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_params} / {num_params}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device != 'cpu' 
    model = model.to(device)
    model.device = device

    def create_optimizer(opt_model, **optimizer_kwargs):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if config.lr_others is not None:
            encoder_parameters = [name for name, _ in opt_model.named_parameters() if "token_rep_layer" in name]
            # encoder_parameters = [name for name, _ in opt_model.token_rep_layer.named_parameters()]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(config.weight_decay_other),
                    "lr": float(config.lr_others),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(config.lr_others),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(config.weight_decay_encoder),
                    "lr": float(config.lr_encoder),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(config.lr_encoder),
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": float(config.weight_decay_encoder),
                    "lr": float(config.lr_encoder),
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": float(config.lr_encoder),
                },
            ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer
    
    optimizer = create_optimizer(model)

    logger.info(f"Using config: \n{json.dumps(config.__dict__, indent=2)}\n\n")

    
    logger.info(f"Checking for duplicate spans and/or relations...")
    for d in [train_data, eval_data]:
        if d:
            for i, item in enumerate(d):
                relation_pos = set()
                for r in item['relations']:
                    position_tuple = (tuple(r['head']['position']), tuple(r['tail']['position']))
                    # assert position_tuple not in relation_pos, f"Duplicate position for relation in (idx {i}) Relation --> {r}"
                    relation_pos.add(position_tuple)

                span_set = set()
                for span in item['ner']:
                    span_pos = (span[0], span[1])
                    # assert span_pos not in span_set, f"Duplicate span in (idx {i}) Span --> {span}"
                    span_set.add(span_pos)

    try:
        train(model, optimizer, train_data=train_data, config=config, train_rel_types=train_rel_types, eval_rel_types=eval_rel_types, eval_data=eval_data,
            num_steps=config.num_steps, eval_every=config.eval_every, top_k=config.top_k,
            log_dir=config.log_dir, wandb_log=args.wandb_log, wandb_sweep=args.wandb_sweep, warmup_ratio=config.warmup_ratio, train_batch_size=config.train_batch_size,
            device=device, use_amp=use_amp)
    except EarlyStoppingException:
        logger.info("Early stopping triggered.")


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    config = load_config_as_namespace(args.config)

    assert not (args.wandb_log is True and args.wandb_sweep is True), "Cannot use both wandb logging and wandb sweep at the same time."

    if args.wandb_sweep:

        if args.experiment:
            if config.dataset_name.lower() == 'redocred':
                sweep_configuration = REDOCRED_EXP_SWEEP_CONFIG
            else:
                sweep_configuration = EXP_SWEEP_CONFIG
        else:
            sweep_configuration = HP_SWEEP_CONFIG

        sweep_configuration["method"] = args.sweep_method  # https://docs.wandb.ai/guides/sweeps/sweep-config-keys#method
        # get day and time as string
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
        sweep_name = f"sweep-{dt_string}"
        sweep_configuration["name"] = sweep_name


        # Initialize sweep by passing in config
        project = "Pair2Rel"
        if args.sweep_id:
            logger.info(f"Resuming sweep with ID: {args.sweep_id}")
            sweep_id = args.sweep_id
        else:
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

        # Start sweep job
        wandb.agent(sweep_id, function=partial(main, args), count=100, project=project)
    else:
        main(args)
