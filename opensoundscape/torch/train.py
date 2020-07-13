#!/usr/bin/env python3
import torch
import torch.nn as nn
from opensoundscape.datasets import BinaryFromAudio
from opensoundscape.metrics import Metrics
import opensoundscape.torch.spec_augment as augment


def train_binary(
    model,
    train_df,
    valid_df,
    optimizer,
    loss_fn,
    epochs=25,
    batch_size=1,
    num_workers=0,
    log_every=5,
    spec_augment=False,
    debug=False,
):
    """ Train a model

    Input:
        model:          A binary torch model, e.g. torchvision.models.resnet18(pretrained=True)
                        - must override classes, e.g. model.fc = torch.nn.Linear(model.fc.in_features, 2)
        train_df:       The training DataFrame with columns "Destination" and "NumericLabels"
        valid_df:       The validation DataFrame with columns "Destination" and "NumericLabels"
        optimize:       A torch optimizer, e.g. torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn:        A torch loss function, e.g. torch.nn.CrossEntropyLoss()
        epochs:         The number of epochs [default: 25]
        batch_size:     The size of the batches [default: 1]
        num_workers:    The number of cores to use for batch preparation [default: 1]
        log_every:      Log statistics when epoch % log_every == 0 [default: 5]
        spec_augment:   Whether or not to use the spec_augment procedure [default: False]
        debug:          Whether or not to write intermediate images [default: False]

    Output:
        A list of dictionaries with keys:
            - epoch
            - train_loss
            - train_accuracy
            - train_precision
            - train_recall
            - train_f1
            - valid_accuracy
            - valid_precision
            - valid_recall
            - valid_f1
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_dataset = BinaryFromAudio(train_df, spec_augment=spec_augment, debug=debug)
    valid_dataset = BinaryFromAudio(valid_df, spec_augment=spec_augment, debug=debug)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.to(device)

    stats = []
    for epoch in range(epochs):
        train_metrics = Metrics(2)
        model.train()
        for t in train_loader:
            X, y = t["X"], t["y"]
            X.to(device)
            y.to(device)
            targets = y.squeeze(1)

            if spec_augment:
                X = augment.time_warp(X.clone(), W=10)
                X = augment.time_mask(X, T=50, max_masks=5)
                X = augment.freq_mask(X, F=50, max_masks=5)
                X = torch.cat([X] * 3, dim=1)

            outputs = model(X)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics.update_loss(loss.clone().detach().item())
            predictions = outputs.clone().detach().argmax(dim=1)
            train_metrics.update_metrics(targets, predictions)

        valid_metrics = Metrics(2)
        model.eval()
        with torch.no_grad():
            for t in valid_loader:
                X, y = t["X"], t["y"]
                X.to(device)
                y.to(device)
                targets = y.squeeze(1)

                if spec_augment:
                    X = torch.cat([X.clone()] * 3, dim=1)

                outputs = model(X)
                predictions = outputs.clone().detach().argmax(dim=1)
                valid_metrics.update_metrics(targets, predictions)

        if epoch % log_every == 0:
            t_loss, t_acc, t_prec, t_rec, t_f1 = train_metrics.compute_metrics(
                len(train_loader)
            )
            _, v_acc, v_prec, v_rec, v_f1 = valid_metrics.compute_metrics(
                len(valid_loader)
            )

            stats.append(
                {
                    "epoch": epoch,
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
            )

    return stats
