"""
Train a Deep Learning model (TensorFlow/Keras) for drug-food interaction severity prediction.

FIXED VERSION: Uses LabelEncoder instead of StringLookup to avoid dtype issues.

Dataset: data/drug_food_interactions.csv (1037+ drug-food pairs)
Severity: continuous 0.0-5.0 scale
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =====================================================================
# PATHS & CONFIGURATION
# =====================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

INTERACTIONS_PATH = DATA_DIR / "drug_food_interactions.csv"
FOOD_KB_PATH = DATA_DIR / "food_to_ingredient_kb.json"
OUTPUT_MODEL_PATH = MODEL_DIR / "jivara_tf_reasoning.keras"
ENCODER_PATH = MODEL_DIR / "drug_category_encoder.pkl"

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42


# =====================================================================
# 1. PREPROCESSING: Load & Prepare Data
# =====================================================================

def load_ingredients_kb() -> dict:
    """Load food_to_ingredient_kb.json"""
    if not FOOD_KB_PATH.exists():
        raise FileNotFoundError(f"Food KB not found: {FOOD_KB_PATH}")

    with open(FOOD_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("food_to_ingredients", {})


def load_interactions_data() -> pd.DataFrame:
    """Load drug_food_interactions.csv"""
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Interactions data not found: {INTERACTIONS_PATH}")

    df = pd.read_csv(INTERACTIONS_PATH)
    print(f"[OK] Loaded {len(df)} drug-food interaction records")
    print(f"Columns: {list(df.columns)}")
    return df


def build_ingredient_vocabulary(food_to_ingredients: dict) -> list[str]:
    """Extract all unique ingredients and return sorted list"""
    all_ingredients = set()
    for ingredients in food_to_ingredients.values():
        all_ingredients.update(ingredients)

    vocab = sorted(list(all_ingredients))
    print(f"[OK] Built ingredient vocabulary: {len(vocab)} unique ingredients")
    return vocab


def create_multihotencoded_ingredients(
    food_class: str,
    food_to_ingredients: dict,
    ingredient_vocab: list[str],
) -> np.ndarray:
    """Convert food class to multi-hot encoded vector"""
    ingredients = food_to_ingredients.get(food_class, [])

    encoded = np.zeros(len(ingredient_vocab), dtype=np.float32)

    for ing in ingredients:
        if ing in ingredient_vocab:
            idx = ingredient_vocab.index(ing)
            encoded[idx] = 1.0

    return encoded


def preprocess_data(
    df: pd.DataFrame,
    food_to_ingredients: dict,
    ingredient_vocab: list[str],
) -> tuple:
    """
    Preprocess: Extract features and target.

    Returns:
        (drug_categories_encoded, ingredient_multihots, severity_targets, label_encoder)
    """
    drug_categories = []
    ingredient_multihots = []
    severity_targets = []

    for idx, row in df.iterrows():
        drug_cat = row.get("drug_category", "unknown")
        food_cls = row.get("food_class", "unknown")
        severity = row.get("severity", 0.0)

        # Skip invalid rows
        if pd.isna(severity) or pd.isna(drug_cat) or pd.isna(food_cls):
            continue

        drug_categories.append(str(drug_cat).strip())
        ingredient_multihots.append(
            create_multihotencoded_ingredients(
                food_cls, food_to_ingredients, ingredient_vocab)
        )
        severity_targets.append(float(severity))

    # Encode drug categories using LabelEncoder
    label_encoder = LabelEncoder()
    drug_categories_arr = label_encoder.fit_transform(drug_categories)
    drug_categories_arr = drug_categories_arr.astype(np.int32)

    ingredient_multihots_arr = np.array(ingredient_multihots, dtype=np.float32)
    severity_targets_arr = np.array(
        severity_targets, dtype=np.float32).reshape(-1, 1)

    print(f"[OK] Preprocessed data:")
    print(f"    Drug categories (encoded): {drug_categories_arr.shape}")
    print(f"    Ingredient multihots: {ingredient_multihots_arr.shape}")
    print(f"    Severity targets: {severity_targets_arr.shape}")
    print(f"    Unique drug categories: {len(label_encoder.classes_)}")

    return drug_categories_arr, ingredient_multihots_arr, severity_targets_arr, label_encoder


# =====================================================================
# 2. CUSTOM COMPONENTS
# =====================================================================

# =====================================================================
# 2. CUSTOM COMPONENTS
# =====================================================================

def asymmetric_severity_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function: asymmetric penalty on underprediction.

    If y_pred < y_true (model underestimates severity): multiply by 3.0
    Otherwise: multiply by 1.0
    """
    error = y_true - y_pred
    squared_error = tf.square(error)

    penalty = tf.where(
        y_pred < y_true,
        3.0,
        1.0
    )

    weighted_loss = squared_error * penalty
    return tf.reduce_mean(weighted_loss)


class TargetMAECallback(tf.keras.callbacks.Callback):
    """Stop training when val_mae <= 0.02"""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_mae = logs.get("val_mae")

        if val_mae is not None and val_mae <= 0.02:
            print(
                f"\n[SUCCESS] Epoch {epoch + 1}: "
                f"val_mae={val_mae:.6f} <= 0.02 target reached! "
                f"Training stopped."
            )
            self.model.stop_training = True
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: val_mae={val_mae:.6f}")


# =====================================================================
# 3. BUILD TENSORFLOW MODEL (FUNCTIONAL API)
# =====================================================================

def build_tf_model(
    num_drug_categories: int,
    num_ingredients: int,
) -> tf.keras.Model:
    """
    Build functional Keras model for severity prediction.

    Architecture:
    - drug_input: Integer → Embedding(64) → Flatten
    - ingredient_input: multi-hot vector (num_ingredients,)
    - Concatenate both
    - Dense(128) + ReLU + Dropout(0.3)
    - Dense(64) + ReLU + Dropout(0.2)
    - Dense(32) + ReLU + Dropout(0.2)
    - Dense(1, linear) → severity output [0.0, 5.0]
    """

    # Input layer 1: Drug categories (integer)
    drug_input = tf.keras.Input(
        shape=(1,), dtype=tf.int32, name="drug_category_input")

    # Embedding layer
    drug_embedding = tf.keras.layers.Embedding(
        input_dim=num_drug_categories + 1,
        output_dim=64,
        name="drug_embedding"
    )(drug_input)

    # Flatten embedding
    drug_flat = tf.keras.layers.Flatten(name="drug_flatten")(drug_embedding)

    # Input layer 2: Multi-hot encoded ingredients
    ingredient_input = tf.keras.Input(
        shape=(num_ingredients,),
        dtype=tf.float32,
        name="ingredient_input"
    )

    # Concatenate both inputs
    merged = tf.keras.layers.Concatenate(name="merge_inputs")(
        [drug_flat, ingredient_input]
    )

    # Dense layers with Dropout
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(merged)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

    x = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_3")(x)

    # Output layer: severity score [0, 5]
    severity_output = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="severity_output"
    )(x)

    # Create model
    model = tf.keras.Model(
        inputs=[drug_input, ingredient_input],
        outputs=severity_output,
        name="JivaraTFReasoning"
    )

    return model


# =====================================================================
# 4. MAIN TRAINING PIPELINE
# =====================================================================

def main():
    """Main training pipeline"""

    print("\n" + "="*70)
    print("JIVARA AI - TENSORFLOW DEEP LEARNING MODEL TRAINING (FIXED)")
    print("="*70 + "\n")

    # Step 1: Load data
    print("[STEP 1] Loading data...")
    food_to_ingredients = load_ingredients_kb()
    df_interactions = load_interactions_data()

    # Step 2: Build vocabulary
    print("\n[STEP 2] Building ingredient vocabulary...")
    ingredient_vocab = build_ingredient_vocabulary(food_to_ingredients)

    # Step 3: Preprocess data
    print("\n[STEP 3] Preprocessing data...")
    drug_cats_encoded, ingredient_mh, severity_targets, label_encoder = preprocess_data(
        df_interactions,
        food_to_ingredients,
        ingredient_vocab,
    )

    # Step 4: Train-test split
    print("\n[STEP 4] Splitting data...")
    (X_drug_train, X_drug_val,
     X_ing_train, X_ing_val,
     y_train, y_val) = train_test_split(
        drug_cats_encoded,
        ingredient_mh,
        severity_targets,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
    )

    print(f"Training set: {len(X_drug_train)} samples")
    print(f"Validation set: {len(X_drug_val)} samples")

    # Step 5: Build model
    print("\n[STEP 5] Building TensorFlow model...")
    num_unique_drugs = len(label_encoder.classes_)
    num_ingredients = len(ingredient_vocab)

    model = build_tf_model(
        num_drug_categories=num_unique_drugs,
        num_ingredients=num_ingredients,
    )

    # Step 6: Compile model
    print("\n[STEP 6] Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=asymmetric_severity_loss,
        metrics=["mae"],
    )

    print("\nModel Summary:")
    model.summary()

    # Step 7: Reshape for integer input
    print("\n[STEP 7] Preparing training data...")
    X_drug_train_reshaped = X_drug_train.reshape(-1, 1)
    X_drug_val_reshaped = X_drug_val.reshape(-1, 1)

    # Step 8: Train model
    print("\n[STEP 8] Training model...")
    print("-" * 70)

    history = model.fit(
        x=[X_drug_train_reshaped, X_ing_train],
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([X_drug_val_reshaped, X_ing_val], y_val),
        callbacks=[TargetMAECallback()],
        verbose=1,
    )

    # Step 9: Evaluate on validation set
    print("\n[STEP 9] Final Evaluation on Validation Set:")
    print("-" * 70)
    val_loss, val_mae = model.evaluate(
        [X_drug_val_reshaped, X_ing_val],
        y_val,
        verbose=0,
    )
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")

    # Step 10: Save model and encoder
    print("\n[STEP 10] Saving model and encoder...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_MODEL_PATH))
    print(f"[OK] Model saved to: {OUTPUT_MODEL_PATH}")

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[OK] Encoder saved to: {ENCODER_PATH}")

    # Step 11: Summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total training samples: {len(X_drug_train)}")
    print(f"Total validation samples: {len(X_drug_val)}")
    print(f"Number of drug categories: {num_unique_drugs}")
    print(f"Number of ingredients: {num_ingredients}")
    print(f"Final Validation MAE: {val_mae:.6f}")
    print(f"Model saved to: {OUTPUT_MODEL_PATH}")
    print(f"Encoder saved to: {ENCODER_PATH}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
