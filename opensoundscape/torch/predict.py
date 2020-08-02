#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.functional import softmax
import yaml


def predict(
    model,
    prediction_dataset,
    batch_size=1,
    num_workers=0,
    apply_softmax=False,
    label_dict=None,
):
    """ Generate predictions on a dataset from a pytorch model object

    Input:
        model:          A binary torch model, e.g. torchvision.models.resnet18(pretrained=True)
                        - must override classes, e.g. model.fc = torch.nn.Linear(model.fc.in_features, 2)
        prediction_dataset: 
                        a pytorch dataset object that returns tensors, such as datasets.SingleTargetAudioDataset()                
        batch_size:     The size of the batches (# files) [default: 1]
        num_workers:    The number of cores to use for batch preparation [default: 0]
                        - if 0, it uses the current process rather than creating a new one
        apply_softmax:  Apply a softmax activation layer to the raw outputs of the model
        label_dict:     List of names of each class, with indices corresponding to NumericLabels [default: None]
                        - if None, the dataframe returned will have numeric column names
                        - if list of class names, returned dataframe will have class names as column names

    Output:
        A dataframe with the CNN prediction results for each class and each file
        
    Notes:
        if label_dict is not None, the returned dataframe's columns will be class names instead of numeric labels
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        prediction_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # run prediction
    all_predictions = []
    for i, inputs in enumerate(dataloader):
        predictions = model(inputs["X"].to(device))
        if apply_softmax:
            softmax_val = softmax(predictions, 1).detach().cpu().numpy()
            for x in softmax_val:
                all_predictions.append(x)
        else:
            for x in predictions.detach().cpu().numpy():
                all_predictions.append(list(x))  # .astype('float64')

    img_paths = prediction_dataset.df[prediction_dataset.filename_column].values
    pred_df = pd.DataFrame(index=img_paths, data=all_predictions)

    if label_dict is not None:
        pred_df = pred_df.rename(columns=label_dict)

    return pred_df
