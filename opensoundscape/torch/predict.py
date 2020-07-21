#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
from torch.nn.functional import softmax

from opensoundscape.datasets import BinaryFromAudio

def predict(
    model,
    prediction_dataset,
    batch_size=1,
    num_workers=0,
    apply_softmax=False,
):
    """ Generate predictions on a dataset from a pytorch model object

    Input:
        save_dir:       A directory to save intermediate weights
        model:          A binary torch model, e.g. torchvision.models.resnet18(pretrained=True)
                        - must override classes, e.g. model.fc = torch.nn.Linear(model.fc.in_features, 2)
        prediction_dataset: 
                        a pytorch dataset object that returns tensors, such as datasets.PredictionDataset()                
        batch_size:     The size of the batches (# files) [default: 1]
        num_workers:    The number of cores to use for batch preparation [default: 0]
                        - if 0, it uses the current process rather than creating a new one
        apply_softmax:  Apply a softmax activation layer to the raw outputs of the model

    Output:
        A dataframe with the CNN prediction results for each class and each file
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        prediction_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # run prediction
    all_predictions = []
    for i, inputs in enumerate(dataloader):
        predictions = model(inputs['X'])
        if apply_softmax:
            softmax_val = softmax(predictions, 1).detach().cpu().numpy()
            for x in softmax_val:
                all_predictions.append(x)
        else:
            for x in predictions.detach().numpy():
                all_predictions.append(
                    list(x)
                )  # .astype('float64')

    # how do we get class names? are they saved in the file?
    # the column names of the output should be class names
    #@BarryMoore re-add the class labels here
    img_paths = prediction_dataset.df[prediction_dataset.audio_column].values
    return pd.DataFrame(index=img_paths, data=all_predictions)