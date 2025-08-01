#!/usr/bin/env python3
"""
Debug script to trace early stopping step counting
"""
import torch
from opensoundscape.ml import shallow_classifier
import opensoundscape as opso

# Patch the quick_fit function to add debug prints
original_quick_fit = shallow_classifier.fit


def debug_quick_fit(*args, **kwargs):
    print(
        "Debug: Calling quick_fit with early_stopping_patience =",
        kwargs.get("early_stopping_patience"),
    )
    return original_quick_fit(*args, **kwargs)


shallow_classifier.fit = debug_quick_fit

print("=== Debug: Early stopping step counting ===")

# Simple test case
mlp = opso.MLPClassifier(3, 1, hidden_layer_sizes=(4,))
train_features = torch.randn(20, 3)
train_labels = torch.randint(0, 2, (20, 1)).float()
val_features = torch.randn(8, 3)
val_labels = torch.randint(0, 2, (8, 1)).float()

print("\nTest: validation_interval=2, early_stopping_patience=4")
print("Expected: Validation at steps 2, 4, 6, 8, 10, ...")
print("If no improvement after step 2, should stop at step 6 (4 steps later)")

shallow_classifier.fit(
    mlp,
    train_features,
    train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    batch_size=8,
    steps=15,
    validation_interval=2,
    early_stopping_patience=4,
    logging_interval=1,
)

print("Debug test completed!")
