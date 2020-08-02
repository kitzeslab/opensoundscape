#!/usr/bin/env python3
import torch
import torch.nn as nn
from opensoundscape.datasets import SingleTargetAudioDataset
from opensoundscape.metrics import Metrics
import opensoundscape.torch.tensor_augment as tensaug
import yaml
from os import path
import time


def train(
    save_dir,
    model,
    train_dataset,
    valid_dataset,
    optimizer,
    loss_fn,
    epochs=25,
    batch_size=1,
    num_workers=0,
    log_every=5,
    tensor_augment=False,
    debug=False,
    print_logging=True,
):
    """ Train a model

    Input:
        save_dir:       A directory to save intermediate results
        model:          A binary torch model,
                        - e.g. torchvision.models.resnet18(pretrained=True)
                        - must override classes, e.g. model.fc = torch.nn.Linear(model.fc.in_features, 2)
        train_dataset:  The training Dataset, e.g. created by SingleTargetAudioDataset()
        valid_dataset:  The validation Dataset, e.g. created by SingleTargetAudioDataset()
        optimizer:       A torch optimizer, e.g. torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn:        A torch loss function, e.g. torch.nn.CrossEntropyLoss()
        epochs:         The number of epochs [default: 25]
        batch_size:     The size of the batches [default: 1]
        num_workers:    The number of cores to use for batch preparation [default: 1]
        log_every:      Log statistics when epoch % log_every == 0 [default: 5]
        tensor_augment: Whether or not to use the tensor augment procedures [default: False]
        debug:          Whether or not to write intermediate images [default: False]

    Side Effects:
        Write a file `epoch-{epoch}.tar` containing (rate of `log_every`):
        - Model state dictionary
        - Optimizer state dictionary
        - Labels in YAML format
        - Train: loss, accuracy, precision, recall, and f1 score
        - Validation: accuracy, precision, recall, and f1 score
        - train_dataset.label_dict
        Write a metadata file with parameter values to save_dir/metadata.txt
    Output:
        None
        
    Effects:
        model parameters are saved to 
        
    """
    if save_dir is not None:
        # save model parameters to metadata file
        metadata = {
            "training_start_time": time.strftime("%X %x %Z"),
            "train_data_len": len(train_dataset),
            "valid_data_len": len(valid_dataset),
            "optimizer": str(optimizer),
            "loss_fn": str(loss_fn),
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "log_every": log_every,
            "tensor_augment": tensor_augment,
            "debug": debug,
            "cuda_is_available:": torch.cuda.is_available(),
        }

        with open(path.join(save_dir, "metadata.txt"), "w") as f:
            f.writelines(yaml.dump(metadata))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    labels_yaml = yaml.dump(train_dataset.label_dict)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.to(device)

    # Model training
    stats = []
    for epoch in range(epochs):
        if print_logging:
            print(f"Epoch {epoch}")
            print("  Training.")
        train_metrics = Metrics(model.fc.out_features)
        model.train()
        for t in train_loader:
            X, y = t["X"], t["y"]
            X = X.to(device)
            y = y.to(device)
            targets = y.squeeze(1)

            if tensor_augment:
                # X is currently shape [batch_size, 3, width, height]
                # Take to shape [batch_size, 1, width, height] for use with `augment`
                X = X[:, 0].unsqueeze(1)
                X = tensaug.time_warp(X.clone(), W=10)
                X = tensaug.time_mask(X, T=50, max_masks=5)
                X = tensaug.freq_mask(X, F=50, max_masks=5)

                # Take from 1 dimension to 3 dimensions
                X = torch.cat([X] * 3, dim=1)

            outputs = model(X)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics.update_loss(loss.clone().detach().item())
            predictions = outputs.clone().detach().argmax(dim=1)
            train_metrics.update_metrics(targets.cpu(), predictions.cpu())

        if print_logging:
            print("  Validating.")
        valid_metrics = Metrics(model.fc.out_features)
        model.eval()
        with torch.no_grad():
            for t in valid_loader:
                X, y = t["X"], t["y"]
                X = X.to(device)
                y = y.to(device)
                targets = y.squeeze(1)

                outputs = model(X)
                predictions = outputs.clone().detach().argmax(dim=1)
                valid_metrics.update_metrics(targets.cpu(), predictions.cpu())

        # Save weights at every logging interval and at the last epoch
        if (epoch % log_every == 0) or (epoch == epochs - 1):
            t_loss, t_acc, t_prec, t_rec, t_f1 = train_metrics.compute_metrics(
                len(train_loader)
            )
            _, v_acc, v_prec, v_rec, v_f1 = valid_metrics.compute_metrics(
                len(valid_loader)
            )

            epoch_results = {
                "train_loss": t_loss,
                "train_accuracy": t_acc,
                "train_precision": t_prec,
                "train_recall": t_rec,
                "train_f1": t_f1,
                "valid_accuracy": v_acc,
                "valid_precision": v_prec,
                "valid_recall": v_rec,
                "valid_f1": v_f1,
            }

            if print_logging:
                print("  Validation results:")
                for metric, result in epoch_results.items():
                    print(f"    {metric}: {result}")

            epoch_results.update(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "labels_yaml": labels_yaml,
                }
            )

            if save_dir is not None:
                epoch_filename = f"{save_dir}/epoch-{epoch}.tar"
                torch.save(epoch_results, epoch_filename)
                if print_logging:
                    print(f"  Saved results to {epoch_filename}.")

    print("Training complete.")
    return
