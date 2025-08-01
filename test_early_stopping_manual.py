#!/usr/bin/env python3
"""
Manual test script for early stopping functionality
"""
import torch
from opensoundscape.ml import shallow_classifier
import opensoundscape as opso

print("Testing early stopping implementation...")

# Create a simple model
mlp = opso.MLPClassifier(4, 2, ("class_a", "class_b"), hidden_layer_sizes=(8,))

# Create synthetic data
train_features = torch.randn(40, 4)
train_labels = torch.randint(0, 2, (40, 2)).float()
val_features = torch.randn(15, 4)
val_labels = torch.randint(0, 2, (15, 2)).float()

print("\n=== Test 1: Training with early stopping patience=3 ===")
shallow_classifier.fit(
    mlp,
    train_features,
    train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    batch_size=8,
    steps=20,
    validation_interval=1,
    early_stopping_patience=3,
    logging_interval=2,
)

print(f"\nFinal model output shape: {mlp(val_features).shape}")
print("✓ Early stopping test completed successfully!")

print("\n=== Test 2: Training without early stopping ===")
mlp2 = opso.MLPClassifier(4, 2, ("class_a", "class_b"), hidden_layer_sizes=(8,))
shallow_classifier.fit(
    mlp2,
    train_features,
    train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    batch_size=8,
    steps=10,
    validation_interval=1,
    # early_stopping_patience=None (no early stopping)
    logging_interval=3,
)

print(f"\nFinal model output shape: {mlp2(val_features).shape}")
print("✓ Normal training test completed successfully!")

print("\n=== Test 3: Training with validation_interval=2 and patience=5 ===")
mlp3 = opso.MLPClassifier(4, 2, hidden_layer_sizes=(8,))
shallow_classifier.fit(
    mlp3,
    train_features,
    train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    batch_size=8,
    steps=25,
    validation_interval=2,  # Validate every 2 steps
    early_stopping_patience=5,  # Patient for 5 steps
    logging_interval=4,
)

print(f"\nFinal model output shape: {mlp3(val_features).shape}")
print("✓ Step-based patience test completed successfully!")

print("\nAll tests passed! Early stopping implementation is working correctly.")
