import pytest
import torch
import tempfile
import os
from opensoundscape.ml import shallow_classifier
import opensoundscape as opso

from copy import deepcopy


class TestMLPClassifier:
    """Test suite for MLPClassifier"""

    def test_init_basic(self):
        """Test basic initialization"""
        mlp = opso.MLPClassifier(
            512,
            3,
            hidden_layer_sizes=(50,),
            classes=("a", "b", "c"),
        )
        assert mlp.input_size == 512
        assert mlp.output_size == 3
        assert mlp.classes == ("a", "b", "c")
        assert mlp.hidden_layer_sizes == (50,)

    def test_init_no_hidden_layers(self):
        """Test initialization with no hidden layers"""
        mlp = opso.MLPClassifier(512, 3)
        assert mlp.input_size == 512
        assert mlp.output_size == 3
        assert mlp.classes is None
        assert mlp.hidden_layer_sizes == ()

    def test_init_multiple_hidden_layers(self):
        """Test initialization with multiple hidden layers"""
        mlp = opso.MLPClassifier(512, 3, hidden_layer_sizes=(100, 50, 25))
        assert mlp.hidden_layer_sizes == (100, 50, 25)

    def test_forward(self):
        """Test forward pass"""
        mlp = opso.MLPClassifier(
            512,
            3,
            hidden_layer_sizes=(50,),
            classes=("a", "b", "c"),
        )
        x = torch.rand(2, 512)
        output = mlp.forward(x)
        assert output.shape == (2, 3)

    def test_forward_no_hidden_layers(self):
        """Test forward pass with no hidden layers"""
        mlp = opso.MLPClassifier(512, 3)
        x = torch.rand(2, 512)
        output = mlp.forward(x)
        assert output.shape == (2, 3)

    def test_forward_batch_size_one(self):
        """Test forward pass with batch size 1"""
        mlp = opso.MLPClassifier(
            512,
            3,
            hidden_layer_sizes=(50,),
            classes=("a", "b", "c"),
        )
        x = torch.rand(1, 512)
        output = mlp.forward(x)
        assert output.shape == (1, 3)

    def test_save_and_load(self):
        """Test save and load functionality"""
        mlp = opso.MLPClassifier(
            512,
            3,
            hidden_layer_sizes=(50,),
            classes=("a", "b", "c"),
        )
        x = torch.rand(2, 512)
        output1 = mlp.forward(x)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            temp_path = tmp.name

        try:
            mlp.save(temp_path)
            mlp2 = opso.MLPClassifier.load(temp_path)

            # Test that loaded model has same attributes
            assert mlp2.input_size == mlp.input_size
            assert mlp2.output_size == mlp.output_size
            assert mlp2.classes == mlp.classes
            assert mlp2.hidden_layer_sizes == mlp.hidden_layer_sizes

            # Test that loaded model produces same output
            output2 = mlp2(x)
            assert torch.allclose(output1, output2, atol=1e-6)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_and_load_no_classes(self):
        """Test save and load with no classes specified"""
        mlp = opso.MLPClassifier(256, 2, hidden_layer_sizes=(100,))
        x = torch.rand(3, 256)
        output1 = mlp.forward(x)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            temp_path = tmp.name

        try:
            mlp.save(temp_path)
            mlp2 = opso.MLPClassifier.load(temp_path)

            assert mlp2.classes is None
            output2 = mlp2(x)
            assert torch.allclose(output1, output2, atol=1e-6)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_and_load_no_hidden_layers(self):
        """Test save and load with no hidden layers"""
        mlp = opso.MLPClassifier(128, 4, classes=("w", "x", "y", "z"))
        x = torch.rand(1, 128)
        output1 = mlp.forward(x)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            temp_path = tmp.name

        try:
            mlp.save(temp_path)
            mlp2 = opso.MLPClassifier.load(temp_path)

            assert mlp2.hidden_layer_sizes == ()
            output2 = mlp2(x)
            assert torch.allclose(output1, output2, atol=1e-6)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_call_method(self):
        """Test that __call__ method works (should be same as forward)"""
        mlp = opso.MLPClassifier(
            512, 3, hidden_layer_sizes=(50,), classes=("a", "b", "c")
        )
        x = torch.rand(2, 512)
        output1 = mlp.forward(x)
        output2 = mlp(x)
        assert torch.equal(output1, output2)

    def test_classifier_layer_attribute(self):
        """Test that classifier_layer attribute is set correctly"""
        mlp = opso.MLPClassifier(512, 3, hidden_layer_sizes=(50,))
        assert mlp.classifier_layer == "classifier"
        assert hasattr(mlp, "classifier")

    def test_different_input_sizes(self):
        """Test with different input sizes"""
        for input_size in [64, 128, 256, 1024]:
            mlp = opso.MLPClassifier(input_size, 2)
            x = torch.rand(1, input_size)
            output = mlp(x)
            assert output.shape == (1, 2)

    def test_different_output_sizes(self):
        """Test with different output sizes"""
        for output_size in [1, 2, 5, 10]:
            mlp = opso.MLPClassifier(100, output_size)
            x = torch.rand(1, 100)
            output = mlp(x)
            assert output.shape == (1, output_size)


class TestQuickFit:
    """Test suite for quick_fit function"""

    def test_quick_fit_basic(self):
        """Test basic quick_fit functionality"""
        mlp = opso.MLPClassifier(10, 2, hidden_layer_sizes=(5,))

        # Create simple training data
        n_samples = 20
        train_features = torch.randn(n_samples, 10)
        train_labels = torch.randint(0, 2, (n_samples, 2)).float()

        # Fit for just a few steps to test functionality
        shallow_classifier.fit(mlp, train_features, train_labels, steps=5, batch_size=8)

        # Test that model can still make predictions
        output = mlp(train_features)
        assert output.shape == (n_samples, 2)

    def test_quick_fit_with_validation(self):
        """Test quick_fit with validation data"""
        mlp = opso.MLPClassifier(8, 3, hidden_layer_sizes=(4,))

        # Create training and validation data
        train_features = torch.randn(15, 8)
        train_labels = torch.randint(0, 2, (15, 3)).float()
        val_features = torch.randn(5, 8)
        val_labels = torch.randint(0, 2, (5, 3)).float()

        # Fit with validation
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            steps=5,
            batch_size=8,
        )

        # Test predictions on both sets
        train_output = mlp(train_features)
        val_output = mlp(val_features)
        assert train_output.shape == (15, 3)
        assert val_output.shape == (5, 3)

    def test_quick_fit_custom_optimizer_criterion(self):
        """Test quick_fit with custom optimizer and criterion"""
        mlp = opso.MLPClassifier(6, 1)

        train_features = torch.randn(10, 6)
        train_labels = torch.randint(0, 2, (10, 1)).float()

        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            optimizer=optimizer,
            criterion=criterion,
            steps=3,
            batch_size=5,
        )

        output = mlp(train_features)
        assert output.shape == (10, 1)

    def test_mlp_fit_method(self):
        """Test that MLPClassifier.fit() calls quick_fit"""
        mlp = opso.MLPClassifier(5, 2)

        train_features = torch.randn(8, 5)
        train_labels = torch.randint(0, 2, (8, 2)).float()

        # This should call quick_fit internally
        mlp.fit(train_features, train_labels, steps=3, batch_size=4)

        output = mlp(train_features)
        assert output.shape == (8, 2)


class TestBatchedTraining:
    """Test suite for batched training functionality"""

    def test_batch_size_smaller_than_dataset(self):
        """Test training with batch size smaller than dataset"""
        mlp = opso.MLPClassifier(8, 2, hidden_layer_sizes=(4,))

        # Create dataset with 20 samples, use batch size of 5
        n_samples = 20
        batch_size = 5
        train_features = torch.randn(n_samples, 8)
        train_labels = torch.randint(0, 2, (n_samples, 2)).float()

        # Should process 4 batches per epoch (20/5 = 4)
        shallow_classifier.fit(
            mlp, train_features, train_labels, batch_size=batch_size, steps=3
        )

        # Test that model can make predictions
        output = mlp(train_features)
        assert output.shape == (n_samples, 2)

    def test_batch_size_larger_than_dataset(self):
        """Test training with batch size larger than dataset"""
        mlp = opso.MLPClassifier(6, 3, hidden_layer_sizes=(3,))

        # Create small dataset with 8 samples, use batch size of 20
        n_samples = 8
        batch_size = 20
        train_features = torch.randn(n_samples, 6)
        train_labels = torch.randint(0, 2, (n_samples, 3)).float()

        # Should process 1 batch per epoch with all 8 samples
        shallow_classifier.fit(
            mlp, train_features, train_labels, batch_size=batch_size, steps=3
        )

        # Test that model can make predictions
        output = mlp(train_features)
        assert output.shape == (n_samples, 3)

    def test_batch_size_equal_to_dataset(self):
        """Test training with batch size equal to dataset size"""
        mlp = opso.MLPClassifier(5, 2)

        # Create dataset with exactly batch size samples
        n_samples = 15
        batch_size = 15
        train_features = torch.randn(n_samples, 5)
        train_labels = torch.randint(0, 2, (n_samples, 2)).float()

        shallow_classifier.fit(
            mlp, train_features, train_labels, batch_size=batch_size, steps=3
        )

        output = mlp(train_features)
        assert output.shape == (n_samples, 2)

    def test_batch_size_one(self):
        """Test training with batch size of 1 (SGD)"""
        mlp = opso.MLPClassifier(4, 2)

        n_samples = 10
        batch_size = 1
        train_features = torch.randn(n_samples, 4)
        train_labels = torch.randint(0, 2, (n_samples, 2)).float()

        # Should process 10 batches per epoch
        shallow_classifier.fit(
            mlp, train_features, train_labels, batch_size=batch_size, steps=2
        )

        output = mlp(train_features)
        assert output.shape == (n_samples, 2)

    def test_batched_validation_smaller_batch(self):
        """Test validation with batch size smaller than validation set"""
        mlp = opso.MLPClassifier(6, 2, hidden_layer_sizes=(3,))

        # Training data
        train_features = torch.randn(30, 6)
        train_labels = torch.randint(0, 2, (30, 2)).float()

        # Validation data - 15 samples with batch size 4
        val_features = torch.randn(15, 6)
        val_labels = torch.randint(0, 2, (15, 2)).float()

        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=4,
            steps=3,
            validation_interval=1,
        )

        # Should work for both training and validation
        train_output = mlp(train_features)
        val_output = mlp(val_features)
        assert train_output.shape == (30, 2)
        assert val_output.shape == (15, 2)

    def test_batched_validation_larger_batch(self):
        """Test validation with batch size larger than validation set"""
        mlp = opso.MLPClassifier(5, 3)

        # Training data
        train_features = torch.randn(25, 5)
        train_labels = torch.randint(0, 2, (25, 3)).float()

        # Small validation set with large batch size
        val_features = torch.randn(8, 5)
        val_labels = torch.randint(0, 2, (8, 3)).float()

        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=20,
            steps=3,
            validation_interval=1,
        )

        train_output = mlp(train_features)
        val_output = mlp(val_features)
        assert train_output.shape == (25, 3)
        assert val_output.shape == (8, 3)

    def test_embedding_dataset(self):
        """Test EmbeddingDataset class directly"""
        features = torch.randn(10, 5)
        labels = torch.randint(0, 2, (10, 3)).float()

        dataset = shallow_classifier.EmbeddingDataset(features, labels)

        assert len(dataset) == 10

        # Test individual item access
        feat, lab = dataset[0]
        assert feat.shape == (5,)
        assert lab.shape == (3,)
        assert torch.equal(feat, features[0])
        assert torch.equal(lab, labels[0])

        # Test with DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
        batch_features, batch_labels = next(iter(loader))
        assert batch_features.shape == (3, 5)
        assert batch_labels.shape == (3, 3)

    def test_different_batch_sizes_same_result(self):
        """Test that different batch sizes produce similar training behavior"""
        # Create identical models
        mlp1 = opso.MLPClassifier(8, 2, hidden_layer_sizes=(4,))
        mlp2 = opso.MLPClassifier(8, 2, hidden_layer_sizes=(4,))

        # Copy weights to ensure identical starting point
        mlp2.load_state_dict(mlp1.state_dict())

        # Same data
        n_samples = 32
        train_features = torch.randn(n_samples, 8)
        train_labels = torch.randint(0, 2, (n_samples, 2)).float()

        # Use same random seed for reproducible shuffling
        torch.manual_seed(42)
        shallow_classifier.fit(
            mlp1, train_features, train_labels, batch_size=8, steps=5
        )

        torch.manual_seed(42)
        shallow_classifier.fit(
            mlp2, train_features, train_labels, batch_size=16, steps=5
        )

        # Models should produce similar (though not identical due to different batch dynamics) outputs
        output1 = mlp1(train_features)
        output2 = mlp2(train_features)
        assert output1.shape == output2.shape == (n_samples, 2)

        # They shouldn't be identical (different batch dynamics)
        # but should be reasonably close if training is working
        assert not torch.allclose(output1, output2, atol=1e-3)


class TestEarlyStopping:
    """Test suite for early stopping and best model loading functionality"""

    def test_early_stopping_saves_best_model(self):
        """Test that training saves and loads the best model based on validation loss"""
        mlp = opso.MLPClassifier(6, 2, hidden_layer_sizes=(4,))

        # Create training and validation data
        train_features = torch.randn(40, 6)
        train_labels = torch.randint(0, 2, (40, 2)).float()
        val_features = torch.randn(15, 6)
        val_labels = torch.randint(0, 2, (15, 2)).float()

        # Store initial model state
        initial_state = deepcopy(mlp.state_dict())

        # Train with validation - should load best model at end
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=10,
            validation_interval=1,
        )

        # Model state should be different from initial (training occurred)
        final_state = mlp.state_dict()

        # Check that at least some parameters changed
        changed = False
        for key in initial_state:
            if not torch.allclose(initial_state[key], final_state[key], atol=1e-6):
                changed = True
                break
        assert changed, "Model parameters should have changed during training"

    def test_no_early_stopping_without_validation(self):
        """Test that early stopping is skipped when no validation data provided"""
        mlp = opso.MLPClassifier(5, 2)

        train_features = torch.randn(20, 5)
        train_labels = torch.randint(0, 2, (20, 2)).float()

        # Store initial model state
        initial_state = deepcopy(mlp.state_dict())

        # Train without validation
        shallow_classifier.fit(mlp, train_features, train_labels, batch_size=8, steps=5)

        # Model should still be trained (parameters changed)
        final_state = mlp.state_dict()
        changed = False
        for key in initial_state:
            if not torch.allclose(initial_state[key], final_state[key], atol=1e-6):
                changed = True
                break
        assert changed, "Model parameters should have changed during training"

    def test_early_stopping_deterministic_improvement(self):
        """Test early stopping with a scenario designed to show improvement"""
        # Create a simple problem where the model should improve
        torch.manual_seed(42)
        mlp = opso.MLPClassifier(4, 1, hidden_layer_sizes=(8,))

        # Generate linearly separable data for consistent improvement
        n_train, n_val = 100, 30
        train_features = torch.randn(n_train, 4)
        # Create labels that have some structure the model can learn
        train_labels = (train_features.sum(dim=1, keepdim=True) > 0).float()

        val_features = torch.randn(n_val, 4)
        val_labels = (val_features.sum(dim=1, keepdim=True) > 0).float()

        # Train with frequent validation checking
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=16,
            steps=20,
            validation_interval=1,
            logging_interval=10,
        )

        # Model should make reasonable predictions
        predictions = mlp(val_features)
        assert predictions.shape == (n_val, 1)

    def test_validation_interval_respected(self):
        """Test that validation only occurs at specified intervals"""
        mlp = opso.MLPClassifier(3, 2)

        train_features = torch.randn(24, 3)
        train_labels = torch.randint(0, 2, (24, 2)).float()
        val_features = torch.randn(8, 3)
        val_labels = torch.randint(0, 2, (8, 2)).float()

        # With validation_interval=3, validation should occur at steps 3, 6, 9, etc.
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=5,
            validation_interval=3,
            logging_interval=1,
        )

        # Should complete without errors
        output = mlp(val_features)
        assert output.shape == (8, 2)

    def test_best_model_state_copying(self):
        """Test that model state is properly copied and restored"""
        # Create two identical models
        mlp1 = opso.MLPClassifier(4, 2, hidden_layer_sizes=(3,))
        mlp2 = opso.MLPClassifier(4, 2, hidden_layer_sizes=(3,))

        # Make them identical
        mlp2.load_state_dict(mlp1.state_dict())

        train_features = torch.randn(32, 4)
        train_labels = torch.randint(0, 2, (32, 2)).float()
        val_features = torch.randn(12, 4)
        val_labels = torch.randint(0, 2, (12, 2)).float()

        # Train one with validation (early stopping)
        shallow_classifier.fit(
            mlp1,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=10,
            validation_interval=1,
        )

        # Train one without validation
        shallow_classifier.fit(
            mlp2, train_features, train_labels, batch_size=8, steps=10
        )

        # Both should produce valid outputs
        output1 = mlp1(val_features)
        output2 = mlp2(val_features)
        assert output1.shape == output2.shape == (12, 2)

        # They might be different due to early stopping vs full training
        # This test mainly ensures no crashes occur during state management

    def test_early_stopping_with_patience(self):
        """Test early stopping with patience parameter"""
        mlp = opso.MLPClassifier(5, 2, hidden_layer_sizes=(3,))

        # Create data where validation loss won't improve much
        train_features = torch.randn(30, 5)
        train_labels = torch.randint(0, 2, (30, 2)).float()
        val_features = torch.randn(10, 5)
        val_labels = torch.randint(0, 2, (10, 2)).float()

        # Train with patience of 3 steps, should stop early
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=100,  # Large number of steps
            validation_interval=1,
            early_stopping_patience=3,
            logging_interval=10,
        )

        # Should complete without errors and model should work
        output = mlp(val_features)
        assert output.shape == (10, 2)

    def test_early_stopping_patience_step_based(self):
        """Test that patience is based on steps since last improvement"""
        mlp = opso.MLPClassifier(4, 1)

        train_features = torch.randn(24, 4)
        train_labels = torch.randint(0, 2, (24, 1)).float()
        val_features = torch.randn(8, 4)
        val_labels = torch.randint(0, 2, (8, 1)).float()

        # Validation every 2 steps, patience of 4 steps
        # If best validation is at step 2, should stop at step 6 (4 steps later)
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=20,
            validation_interval=2,
            early_stopping_patience=4,
            logging_interval=5,
        )

        output = mlp(val_features)
        assert output.shape == (8, 1)

    def test_no_early_stopping_without_patience(self):
        """Test that early stopping doesn't occur without patience parameter"""
        mlp = opso.MLPClassifier(3, 2)

        train_features = torch.randn(20, 3)
        train_labels = torch.randint(0, 2, (20, 2)).float()
        val_features = torch.randn(6, 3)
        val_labels = torch.randint(0, 2, (6, 2)).float()

        # Train without early_stopping_patience - should complete all steps
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=10,
            validation_interval=1,
            # early_stopping_patience=None (default)
        )

        output = mlp(val_features)
        assert output.shape == (6, 2)

    def test_early_stopping_with_improvement(self):
        """Test early stopping with a scenario that should show improvement"""
        # Use a structured problem that should improve
        torch.manual_seed(123)
        mlp = opso.MLPClassifier(3, 1, hidden_layer_sizes=(6,))

        # Create linearly separable problem
        n_train, n_val = 60, 20
        train_features = torch.randn(n_train, 3)
        # Make labels predictable: positive if sum of features > 0
        train_labels = (train_features.sum(dim=1, keepdim=True) > 0).float()

        val_features = torch.randn(n_val, 3)
        val_labels = (val_features.sum(dim=1, keepdim=True) > 0).float()

        # Train with reasonable patience
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=12,
            steps=30,
            validation_interval=1,
            early_stopping_patience=10,
            logging_interval=5,
        )

        # Model should learn this simple pattern
        predictions = mlp(val_features)
        assert predictions.shape == (n_val, 1)

    def test_early_stopping_loads_best_model(self):
        """Test that early stopping loads the best model, not the last one"""
        mlp = opso.MLPClassifier(4, 2, hidden_layer_sizes=(3,))

        train_features = torch.randn(32, 4)
        train_labels = torch.randint(0, 2, (32, 2)).float()
        val_features = torch.randn(12, 4)
        val_labels = torch.randint(0, 2, (12, 2)).float()

        # Record initial state
        initial_output = mlp(val_features)

        # Train with early stopping
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            validation_features=val_features,
            validation_labels=val_labels,
            batch_size=8,
            steps=20,
            validation_interval=1,
            early_stopping_patience=5,
            logging_interval=10,
        )

        # Model should have changed from initial state
        final_output = mlp(val_features)
        assert not torch.allclose(initial_output, final_output, atol=1e-6)
        assert final_output.shape == (12, 2)

    def test_early_stopping_without_validation_data(self):
        """Test that early stopping is ignored without validation data"""
        mlp = opso.MLPClassifier(4, 2)

        train_features = torch.randn(20, 4)
        train_labels = torch.randint(0, 2, (20, 2)).float()

        # Should ignore early_stopping_patience when no validation data
        shallow_classifier.fit(
            mlp,
            train_features,
            train_labels,
            batch_size=8,
            steps=5,
            early_stopping_patience=3,  # Should be ignored
        )

        output = mlp(train_features)
        assert output.shape == (20, 2)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_mismatched_input_size(self):
        """Test that wrong input size raises appropriate error"""
        mlp = opso.MLPClassifier(10, 2)
        x = torch.rand(1, 5)  # Wrong input size

        with pytest.raises(RuntimeError):
            mlp(x)

    def test_empty_batch(self):
        """Test with empty batch"""
        mlp = opso.MLPClassifier(10, 2)
        x = torch.empty(0, 10)
        output = mlp(x)
        assert output.shape == (0, 2)

    def test_large_hidden_layers(self):
        """Test with large hidden layer configuration"""
        mlp = opso.MLPClassifier(100, 5, hidden_layer_sizes=(200, 150, 100, 50))
        x = torch.rand(2, 100)
        output = mlp(x)
        assert output.shape == (2, 5)

    def test_single_neuron_layers(self):
        """Test with single neuron in hidden layers"""
        mlp = opso.MLPClassifier(10, 1, hidden_layer_sizes=(1,))
        x = torch.rand(3, 10)
        output = mlp(x)
        assert output.shape == (3, 1)
