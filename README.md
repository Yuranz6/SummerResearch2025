# Summer Research Project 2025 @McGill Univ.

A Research Project in Federated Learning. Repo adapted from the seminal work FedFed: https://github.com/tmlr-group/FedFed

## Dataset
eICU: https://eicu-crd.mit.edu/
Initial preprocessing using drug_harmonize.py

## Architetcure:

### Phase 1: Data Loading & Initialization
'''bash
 BasePSManager.__init__()
  ├── _setup_datasets()
  │   └── load_data() → Data_Loader.load_data()
  │       └── federated_medical_split()
  │           └── partition_eicu_data_by_hospital()  
  │               ├── Adaptive hospital selection
  │               ├── Target hospital exclusion (if unseen_hospital_test=True): target hospital becomes global test set
  │               └── Returns: train_clients + excluded_hospital_data
  ├── _setup_clients() → Creates FedAVGClient instances
  └── _setup_server() → Creates FedAVGAggregator

'''

### Phase 2: Training Pipeline
'''bash
  BasePSManager.train()
  ├── Phase 2.1: VAE Training (if VAE=True)
  │   ├── _share_data_step() → Multiple VAE rounds
  │   │   └── For each client: client.train_vae_model()
  │   └── _get_local_shared_data() → Aggregate shared features
  │
  ├── Phase 2.2: Federated Training
  │   └── For each comm_round:
  │       ├── algorithm_train() → FedAVGManager.algorithm_train()
  │       │   └── For each client: client.train() with shared features
  │       ├── aggregator.aggregate() → FedAvg parameter averaging
  │       └── aggregator.test_on_server_for_round() → Validation
  │
  └── Phase 2.3: Results 
      ├── aggregator.save_classifier() → Save trained model
      ├── _save_training_results() → Save training metrics
      └── run_post_training_evaluation()
'''

# Phase 3: Evaluation:
'''bash
  evaluation/
    ├── integration_helper.py     # Main integration interface
    ├── bootstrap_evaluator.py    # Core bootstrap sampling engine
    ├── metrics.py                # AUPRC & medical metrics
    ├── comparison_evaluator.py   
    ├── fedavg_evaluator.py       # FedAvg baseline (no VAE)
    ├── fedprox_evaluator.py      # FedProx baseline (no VAE)  
    ├── fedfed_evaluator.py       # FedFed (VAE + FedAvg)
    ├── visualization.py          # Plotting (boxplots)
    └── run_evaluation.py         # Standalone evaluation script
'''

## Usage:

### Option 1: Integrated Training + Evaluation
set the necessary params in eicu_config.yaml
'''bash
  python main.py --config_file configs/eicu_config.yaml
'''

# Option 2: Standalone Evaluation
'''bash
python evaluation/run_evaluation.py \
      --target_hospital_id 167 \
      --algorithm fedfed \
      --model_path ./output/trained_model.pth \
      --data_path ./output/hospital_167_data.pkl
'''