# todo: save models with extras in similar way to resnet_binary
# (includes train/valid scores/preds)

# adapted from zhmiao
# github.com/zhmiao/BirdMultiLabel/blob/master/src/algorithms/plain_resnet.py

import os
import numpy as np
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# from tqdm import tqdm
import random

import torch
import torch.optim as optim
from sklearn.metrics import jaccard_score, hamming_loss, precision_recall_fscore_support

from opensoundscape.torch.architectures.distreg_resnet_architecture import (
    DistRegResNetClassifier,
)
from opensoundscape.torch.architectures.plain_resnet import PlainResNetClassifier
from opensoundscape.torch.models.utils import BaseModule, get_dataloader


# NOTE: Turning off all logging for now. may want to use logging module in future


class PytorchModel(BaseModule):

    """
    Generic Pytorch Model with training and prediction, flexible architecture.
    """

    name = "PytorchModel"
    net = None
    optimizer = None
    scheduler = None

    def __init__(
        self, architecture, classes
    ):  # train_dataset, valid_dataset, architecture, weights_path="."):
        """if you want to change other parameters,
        simply create the object then modify them
        TODO: should not require train and valid ds for prediction
        maybe you should just provide the classes, then
        give train_ds and valid_ds to model.train()?
        """
        super(PytorchModel, self).__init__()

        # todo: should I initialize self.optimizer?

        self.weights_path = weights_path

        # self.train_dataset = train_dataset
        # self.valid_dataset = valid_dataset
        # self.num_classes = len(train_dataset.labels)
        self.classes = classes  # train_dataset.labels
        print(f"n classes: {len(self.classes)}")

        ### network parameters ###
        self.weights_init = "ImageNet"
        self.prediction_threshold = 0.25
        self.num_layers = 18  # can use 50 for resnet50
        self.class_aware_sampler = False

        ### training parameters ###
        # defaults from https://github.com/zhmiao/BirdMultiLabel/blob/master/configs/XENO/multi_label_reg_10_091620.yaml
        # feature
        self.lr_feature = 0.001
        self.momentum_feature = 0.9
        self.weight_decay_feature = 0.0005
        # classifier
        self.lr_classifier = 0.01
        self.momentum_classifier = 0.9
        self.weight_decay_classifier = 0.0005
        # lr_scheduler
        self.step_size = 10
        self.gamma = 0.1

        #######################################
        # Setup data for training and testing #
        #######################################
        # will pass dataset to train, predict, evaluate()
        # self.trainloader, self.testloader, self.valloader = load_data(args)

        # need to set self.train_class_counts later
        _, self.train_class_counts = self.train_dataset.class_counts_cal()
        print(f"train class counts: {self.train_class_counts}")

        # self.total_steps = int(self.train_class_counts.sum() / args.batch_size)
        # TODO: why is total steps different from len(df)? It seems to be instead
        # equal to number of total positive labels

        # print(f"Architecture supplied by user")
        self.network = architecture

    def init_optimizer(self, optimizer_class=optim.SGD, params_list=None):
        if params_list is None:
            # default params list
            params_list = [
                {
                    "params": self.network.feature.parameters(),
                    "lr": self.lr_feature,
                    "momentum": self.momentum_feature,
                    "weight_decay": self.weight_decay_feature,
                },
                {
                    "params": self.network.classifier.parameters(),
                    "lr": self.lr_classifier,
                    "momentum": self.momentum_classifier,
                    "weight_decay": self.weight_decay_classifier,
                },
            ]
        return optimizer_class(params_list)

    def set_train(self, batch_size, num_workers):
        ###########################
        # Setup cuda and networks #
        ###########################
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # setup network
        # self.main_logger.info("\nGetting {} model.".format(self.args.model_name))
        self.network.to(self.device)

        ######################
        # Optimization setup #
        ######################
        # Setup optimizer parameters for each network component
        # TODO: optimizer needs to be initialized in order to .load_state_dict()

        # Setup optimizer and optimizer scheduler
        self.optimizer = self.init_optimizer()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

        # set up train_loader and valid_loader dataloaders
        # make a dataloader to supply training images from train_dataset
        # eventually should use models.utils.get_dataloader() for cas sampler
        self.train_loader = get_dataloader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            cas_sampler=self.class_aware_sampler,
        )

        # make a dataloader to supply training images from valid_dataset
        self.valid_loader = get_dataloader(
            self.valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            cas_sampler=False,
        )

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        # self.main_logger.info("\nGetting {} model.".format(self.args.model_name))
        # self.main_logger.info("\nLoading from {}".format(self.weights_path))
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # self.network = DistRegResNetClassifier(
        #     name=self.args.model_name,
        #     num_cls=self.args.num_classes,
        #     weights_init=self.weights_path,
        #     num_layers=self.args.num_layers,
        #     init_feat_only=False,
        #     class_freq=self.train_class_counts,
        # )
        self.network.to(self.device)

    def train_epoch(self, epoch):

        self.network.train()

        for batch_idx, item in enumerate(self.train_loader):

            # all augmentation occurs in the Preprocessor (train_loader)
            data, labels = item["X"].to(self.device), item["y"].to(self.device)
            labels = labels.squeeze(1)

            # # log basic adda train info
            N = len(self.train_loader)
            info_str = "Epoch: {} [batch {}/{} ({:.2f}%)] ".format(
                epoch, batch_idx, N, 100 * batch_idx / N
            )

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.network.feature(data)
            logits = self.network.classifier(feats)
            # calculate loss
            loss = self.network.criterion_cls(logits, labels)

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.optimizer.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.optimizer.step()

            ################
            # Save weights #
            ################
            if batch_idx % self.save_interval == 0:
                self.save(self.weights_path)

            ###########
            # Logging #
            ###########
            if batch_idx % self.log_interval == 0:

                tgts = labels.int().detach().cpu().numpy()

                # Threashold prediction
                preds = (
                    (torch.sigmoid(logits) >= self.prediction_threshold)
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Jaccard score and Hamming loss
                jacc = jaccard_score(tgts, preds, average="macro")
                hamm = hamming_loss(tgts, preds)

                # log update info
                info_str += "Jacc: {:0.3f} Hamm: {:0.3f} DistLoss: {:.3f}".format(
                    jacc, hamm, loss.item()
                )
                print(info_str)

        self.scheduler.step()

    def train(
        self,
        train_dataset,
        valid_dataset,
        num_epochs,
        batch_size=1,
        num_workers=1,
        weights_path=".",
        save_interval=1,
        log_interval=10,
    ):

        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.weights_path = weights_path

        self.set_train(batch_size, num_workers)

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Training
            self.train_epoch(epoch)

            # Validation
            print("\nValidation.")
            val_f1 = self.evaluate(self.valid_loader, set_eval=False)  # ,test=False)
            if val_f1 > best_f1:
                self.network.update_best()
                best_f1 = val_f1
                best_epoch = epoch

        print(
            "\nBest Model Appears at Epoch {} with F1 {:.3f}...".format(
                best_epoch, best_f1 * 100
            )
        )
        self.save_weights()

    def evaluate_epoch(self, loader):

        self.network.eval()

        # Get unique classes in the loader and corresponding counts
        # loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        eval_class_counts = loader.dataset.class_counts_cal()
        loader_uni_class = len(self.classes)
        # todo this is another place I may be messing up the class counts

        total_preds = []
        total_tgts = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for item in loader:
                # setup data
                data = item["X"].to(self.device)
                labels = item["y"].to(self.device)
                data.requires_grad = False
                labels.requires_grad = False

                # reshape data if needed
                # data = torch.cat([data] * 3, dim=1)

                # forward
                feats = self.network.feature(data)
                logits = self.network.classifier(feats)

                # Threshold prediction
                preds = (
                    (torch.sigmoid(logits) >= self.prediction_threshold)
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                )
                tgts = labels.int().detach().cpu().numpy()

                total_preds.append(preds)
                total_tgts.append(tgts)

        total_preds = np.concatenate(total_preds, axis=0)
        total_tgts = np.concatenate(total_tgts, axis=0)

        # Record per class precision, recall, and f1
        class_pre, class_rec, class_f1, _ = precision_recall_fscore_support(
            total_tgts, total_preds, average=None, zero_division=0
        )

        eval_info = "{} Per-class evaluation results: \n".format(
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        for i in range(len(class_pre)):
            eval_info += (
                "[Class {} (train counts {})] ".format(
                    i, self.train_class_counts[i]  # [loader_uni_class][i]
                )
                + "Pre: {:.3f}; ".format(class_pre[i] * 100)
                + "Rec: {:.3f}; ".format(class_rec[i] * 100)
                + "F1: {:.3f}; \n".format(class_f1[i] * 100)
            )

        eval_info += (
            "Macro Pre: {:.3f}; ".format(class_pre.mean() * 100)
            + "Macro Rec: {:.3f}; ".format(class_rec.mean() * 100)
            + "Macro F1: {:.3f} \n".format(class_f1.mean() * 100)
        )

        return eval_info, class_f1.mean()

    def evaluate(self, loader, set_eval=True):

        if set_eval:
            self.set_eval()

        eval_info, eval_f1 = self.evaluate_epoch(loader)
        print(eval_info)

        return eval_f1

    def save(self, path=None):
        """save model weights (default location is self.weights_path)"""
        if path is None:
            path = f"{self.weights_path}/epoch-{self.current_epoch}.model"
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)  # .rsplit("/", 1)[0], exist_ok=True)
        print(f"Saving to {path}")
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()  # TODO: is this correct?
                #'loss': loss,
            }
        )
        # self.network.save(path)

    def load(self, path):
        # TODO: test if saving and loading works properly
        """load model and optimizer state dicts from disk

        the object should be saved with model.save()
        which uses torch.save with keys for 'model_state_dict' and 'optimizer_state_dict'

        """
        checkpoint = torch.load(path)

        # load the nn feature weights from the checkpoint
        self.load_state_dict(checkpoint["model_state_dict"])

        # create an optimizer then load the checkpoint state dict
        self.optimizer = self.init_optimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # self.network.load(path)

        # if path is None:
        #     path = f"{self.weights_path}/epoch-{self.current_epoch}.model"
        # path = Path(path)
        # assert path.exists(), f"did not find a file at {path}"

        # # copying this from the __init__:
        # print(f"Building architecture")
        # self.network = DistRegResNetClassifier(
        #     num_cls=len(self.classes),
        #     weights_init=path,
        #     num_layers=self.num_layers,
        #     init_feat_only=True,
        #     class_freq=np.array(self.train_class_counts),
        # )

    def predict(
        self,
        prediction_dataset,
        batch_size=1,
        num_workers=1,
        # apply_softmax=False
    ):
        """Generate predictions on a dataset from a pytorch model object
        Input:
            prediction_dataset:
                            a pytorch dataset object that returns tensors, such as datasets.SingleTargetAudioDataset()
            batch_size:     The size of the batches (# files) [default: 1]
            num_workers:    The number of cores to use for batch preparation [default: 1]
                            - if you want to use all the cores on your machine, set it to 0 (this could freeze your computer)
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
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.network.eval()
        self.network.to(self.device)

        dataloader = torch.utils.data.DataLoader(
            prediction_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # what does pin_memory=True do?
        )

        # run prediction

        total_logits = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for sample in dataloader:
                data = sample["X"].to(self.device)  # is this correct?

                # setup data
                data.requires_grad = False
                # data = torch.cat([data] * 3, dim=1)

                # forward pass of network: feature extratcor + classifier
                feats = self.network.feature(data)
                logits = self.network.classifier(feats)

                total_logits.append(logits.detach().cpu().numpy())  # [:, target_id])

        total_logits = np.concatenate(total_logits, axis=0)

        # all_predictions = []
        # for i, inputs in enumerate(dataloader):
        #     predictions = self.network(inputs["X"].to(self.device))
        # if apply_softmax:
        #     softmax_val = softmax(predictions, 1).detach().cpu().numpy()
        #     for x in softmax_val:
        #         all_predictions.append(x[1])  # keep the present, not absent
        #     labels = prediction_dataset.df.columns.values
        # else:
        #     for x in predictions.detach().cpu().numpy():
        #         all_predictions.append(list(x))  # .astype('float64')
        #     label = prediction_dataset.df.columns.values[0]
        #     labels = [label + "_absent", label + "_present"]

        img_paths = prediction_dataset.df.index.values
        pred_df = pd.DataFrame(index=img_paths, data=total_logits, columns=self.classes)

        return pred_df


class Resnet18Multilabel(
    PytorchModel
):  # TODO: move train_dataset and valid_dataset elsewhere
    def __init__(self, train_dataset, valid_dataset, weights_path="."):
        """if you want to change other parameters,
        simply create the object then modify them
        """
        self.classes = train_dataset.labels
        self.weights_init = "ImageNet"
        _, self.train_class_counts = train_dataset.class_counts_cal()
        print(f"train class counts: {self.train_class_counts}")

        architecture = DistRegResNetClassifier(  # pass architecture as argument
            num_cls=len(self.classes),
            weights_init=self.weights_init,
            num_layers=18,
            init_feat_only=True,
            class_freq=np.array(self.train_class_counts),
        )

        super(Resnet18Multilabel, self).__init__(
            train_dataset, valid_dataset, architecture, weights_path
        )


class Resnet18Binary(
    PytorchModel
):  # TODO: move train_dataset and valid_dataset elsewhere
    def __init__(self, weights_path="."):
        """if you want to change other parameters,
        simply create the object then modify them
        """
        self.weights_init = "ImageNet"

        architecture = PlainResNetClassifier(  # pass architecture as argument
            num_cls=2,
            weights_init=self.weights_init,
            num_layers=18,
            init_feat_only=True,
        )

        super(Resnet18Binary, self).__init__(
            train_dataset, valid_dataset, architecture, weights_path
        )
