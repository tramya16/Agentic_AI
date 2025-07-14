# First install these in your terminal
# pip install tensorflow-macos
# pip install tensorflow-metal
# pip install deepchem
# pip install rdkit-pypi
# pip install scikit-learn
# pip install numpy pandas matplotlib

import deepchem as dc
import os
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from deepchem.models import SklearnModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
import time

print("Starting model training pipeline on Apple M1...")
print("This may take 10-30 minutes depending on dataset size")

# Create models directory
models_dir = "./trained_models"
os.makedirs(models_dir, exist_ok=True)


def morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint using the new RDKit API"""
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits
    )


def featurize_dataset(dataset, featurizer_fn):
    """Featurize a dataset using a custom featurizer function"""
    features = []
    valid_indices = []

    for i, smiles in enumerate(dataset.ids):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features.append(featurizer_fn(mol))
            valid_indices.append(i)

    # Filter targets to match valid molecules
    targets = dataset.y[valid_indices]
    return np.array(features), targets, valid_indices


def train_and_save_model(dataset_name):
    """
    Train a model for the specified dataset and save it
    """
    print(f"\n=== Training {dataset_name} model ===")
    start_time = time.time()

    # Load dataset
    if dataset_name == 'lipophilicity':
        tasks, datasets, transformers = dc.molnet.load_lipo()
    elif dataset_name == 'esol':
        tasks, datasets, transformers = dc.molnet.load_esol()
    elif dataset_name == 'tox21':
        tasks, datasets, transformers = dc.molnet.load_tox21()
    elif dataset_name == 'bbbp':
        tasks, datasets, transformers = dc.molnet.load_bbbp()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train, valid, test = datasets

    print(f"Tasks: {tasks}")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    # Determine model type
    is_classification = dataset_name in ['tox21', 'bbbp']
    mode = 'classification' if is_classification else 'regression'
    model_path = os.path.join(models_dir, dataset_name)
    os.makedirs(model_path, exist_ok=True)

    # Featurize datasets with new MorganGenerator API
    print("Featurizing data with Morgan fingerprints...")

    # Custom featurizer function
    def featurizer_fn(mol):
        return morgan_fingerprint(mol, radius=2, n_bits=2048)

    train_X, train_y, _ = featurize_dataset(train, featurizer_fn)
    test_X, test_y, _ = featurize_dataset(test, featurizer_fn)

    # Create and train model
    if is_classification:
        sklearn_model = RandomForestClassifier(
            n_estimators=150,  # Reduced for M1 efficiency
            max_depth=12,  # Shallower trees
            random_state=42,
            n_jobs=-1,  # Use all performance cores
            max_samples=0.8,  # Faster training
            verbose=1  # Show progress
        )
        metric_name = "ROC-AUC"
    else:
        sklearn_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            max_samples=0.8,
            verbose=1
        )
        metric_name = "R-squared"

    print(f"Training {mode} model...")
    sklearn_model.fit(train_X, train_y)

    # Wrap in DeepChem model
    model = SklearnModel(sklearn_model, model_dir=model_path)

    # Evaluate
    test_pred = model.predict(test_X)

    if is_classification:
        test_score = roc_auc_score(test_y, test_pred)
    else:
        test_score = r2_score(test_y, test_pred)

    print(f"Test {metric_name}: {test_score:.4f}")

    # Save model
    model.save()
    print(f"Model saved to {model_path}")

    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'tasks': tasks,
        'mode': mode,
        'test_score': float(test_score),
        'model_type': 'RandomForest',
        'featurizer': 'MorganFingerprint(radius=2, n_bits=2048)',
        'training_time': round(time.time() - start_time, 1),
        'training_samples': len(train),
        'test_samples': len(test)
    }

    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return model, tasks, featurizer_fn, test_score


# Train all models
logp_model, logp_tasks, logp_featurizer, logp_score = train_and_save_model('lipophilicity')
sol_model, sol_tasks, sol_featurizer, sol_score = train_and_save_model('esol')
tox_model, tox_tasks, tox_featurizer, tox_score = train_and_save_model('tox21')
bbb_model, bbb_tasks, bbb_featurizer, bbb_score = train_and_save_model('bbbp')

print("\n=== Training Complete ===")
print(f"LogP Model - Test R-squared: {logp_score:.4f}")
print(f"Solubility Model - Test R-squared: {sol_score:.4f}")
print(f"Toxicity Model - Test ROC-AUC: {tox_score:.4f}")
print(f"BBB Model - Test ROC-AUC: {bbb_score:.4f}")


# Test prediction function
def test_prediction(smiles="CCO"):
    print(f"\n=== Testing models with SMILES: {smiles} ===")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("Invalid SMILES!")
        return

    # LogP prediction
    logp_feat = np.array([logp_featurizer(mol)])
    logp_pred = logp_model.predict(logp_feat)[0]
    print(f"LogP: {float(logp_pred):.3f}")

    # Solubility prediction
    sol_feat = np.array([sol_featurizer(mol)])
    sol_pred = sol_model.predict(sol_feat)[0]
    print(f"Solubility: {float(sol_pred):.3f}")

    # Toxicity prediction
    tox_feat = np.array([tox_featurizer(mol)])
    tox_preds = tox_model.predict(tox_feat)[0]
    print(f"Tox21 Tasks: {tox_tasks}")
    print(f"Toxicity Predictions: {np.round(tox_preds, 3)}")

    # BBBP prediction
    bbb_feat = np.array([bbb_featurizer(mol)])
    bbb_pred = bbb_model.predict(bbb_feat)[0][0]
    print(f"BBB Penetration Probability: {float(bbb_pred):.3f}")


# Test predictions
test_prediction("CCO")  # Ethanol
test_prediction("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # Caffeine
test_prediction("C1=CC2=C(C3=C(C=CC=N3)C=C2)N=C1")  # Quinoline
test_prediction("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

print(f"\nAll models saved to: {models_dir}")