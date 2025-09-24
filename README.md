# Summer Research Project 2025 @McGill Univ.

A Research Project in Federated Learning. Repo adapted from the seminal work FedFed: https://github.com/tmlr-group/FedFed
Keywords: Machine Learning, Federated Learning, Healthcare, Disentangled Representation Learning, Time-series EHR

Most up-to-date work in Master branch

## Dataset

eICU: https://eicu-crd.mit.edu/
Initial preprocessing using drug_harmonization.py
MIMIC-III: to be added

## Architetcure:

### Phase 1: Data Loading & Initialization

```
 BasePSManager.__init__()
  ├── _setup_datasets()
  │   └── load_data() → Data_Loader.load_data()
  │       └── federated_medical_split()
  │           └── partition_eicu_data_by_hospital()
  │               ├── hospital selection (adaptive)
  │               ├── Target hospital exclusion (only if experiment unseen_hospital_test=True): target hospital becomes global test set
  │               └── Returns: train_clients + excluded_hospital_data
  ├── _setup_clients() → Creates FedAVGClient instances
  └── _setup_server() → Creates FedAVGAggregator

```

### Phase 2: Training Pipeline

```
  BasePSManager.train()
  ├── Phase 2.1: VAE Training
  │   ├── _share_data_step() → Actual VAE training for multiple rounds
  │   │   └── client.train_vae_model()
  │   └── _get_local_shared_data() → Aggregate shared features to create global shared dataset
  │
  ├── Phase 2.2: Federated Training
  │   └── For each comm_round:
  │       ├── algorithm_train() → FedAVGManager.algorithm_train()
  │       │   └── For each client: client.train() with globally shared features
  │       ├── aggregator.aggregate() → FedAvg parameter averaging
  │       └── aggregator.test_on_server_for_round() → Validation
  │
  └── Phase 2.3: Results
      ├── aggregator.save_classifier()
      ├── _save_training_results() → Save training metrics
```

### Phase 3: Evaluation:

```
  evaluation/
    ├── integration_helper.py     # Main integration interface
    ├── bootstrap_evaluator.py    # Core bootstrap sampling engine
    ├── metrics.py                # AUPRC & medical metrics
    ├── comparison_evaluator.py
    ├── centralized_evaluator.py  # centralized training (aggregate dataset from all hospitals)
    ├── fedavg_evaluator.py       # FedAvg baseline
    ├── fedprox_evaluator.py      # FedProx baseline
    ├── fedfed_evaluator.py       # FedFed (VAE + FedAvg)
    ├── visualization.py          # Plotting (boxplots)
    └── run_evaluation.py         # Standalone evaluation script
```

## Usage:

### Option 1: Integrated Training + Evaluation

set the necessary params in eicu_config.yaml

```
  python main.py --config_file configs/eicu_config.yaml
```

## Option 2: Standalone Evaluation

```
python evaluation/run_evaluation.py \
      --target_hospital_id 167 \
      --algorithm fedfed \
      --model_path ./output/trained_model.pth \
      --data_path data/eicu.csv
```
