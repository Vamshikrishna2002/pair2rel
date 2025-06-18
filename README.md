# pair2rel

Pair2Rel: Relation Extraction Model

## 📁 Project Structure

Pair2Rel/
├── Benchmark datasets/ # Raw and preprocessed dataset files (ADE, SciERC, etc.)

├── Evaluation_pipeline/ # Jupyter notebooks for evaluating trained models

├── gpt_4o_mini_evaluation/ # GPT-4o/mini-based evaluation notebooks
├── configs/ # YAML config files for training 
├── CONLL04/, DEVREV/, data/ # Task-specific input or output formatting code
├── lib/ # (Optional) helper library code
├── logs/ # Saved logs from training/evaluation
├── pair2rel/ # Core model logic (model, modules, utils)
│ ├── init.py
│ ├── model.py
│ └── modules/ # entity representation layers, relation representation layers etc.
├── pair2rel_weights/ # Pretrained model weights
├── chatgpt_4o_mini_extracted_named_entities.json # GPT-4o-mini entities output fro benchmark datasets
├── chatgpt_4o_mini_relation_extraction.json # GPT-4o-mini relations output for benchmark datasets
├── pipeline.py # Full pipeline code
├── train.py # Training script
├── run_evaluation.py
├── spacy_help_functions.py # Utilities for spaCy-based preprocessing
├── entities.json # Entity types with label indices
├── requirements.txt # Dependencies
├── requirements-dev.txt # Additional dev dependencies
├── temp.ipynb # Notebook for debugging/experiments
├── README.md # This file

BenchMark Datasets:

CONLL04
SCIERC
NYT
ADE

chatgpt_4o_mini_extracted_named_entities:

This folder contains json files of benchmark datasets where entities were extracted using chatgpt-4o-mini

chatgpt_4o_mini_relation_extraction:

This folder contains json files of benchmark datasets where relations were extracted using chatgpt-4o-mini

data:

Contains training data for the model

DEVREV:

Applied our pipeline on devrev dataset

Evaluation_pipeline:

Evalution of Benchmark datasets using our pipeline. Each jupyter notebook specific to each dataset

gpt_4o_mini_evaluation:

Evalution of Benchmark datasets using chatgpt_4o_mini. Each jupyter notebook specific to each dataset

pair2rel:

It contains modules folder where each file represents for different layers and task in architecture. All these will be used in train.py file

pair2rel_weights:

It contains the weights of the trained model.

train.py:

to train any file 

CUDA_VISIBLE_DEVICES="0" python train.py --config configs/config_conll2004.yaml 

pipeline.py

input sentence, entity labels and relation labels to get output knowledge graph 





