# adapt from Miao's repository github.com/zhmiao/BirdMultiLabel
# this .py file depends on a few other things in the repository:
# 1. selecting a "model" from the models folder, where models are hand-built architectures+loss funcitons
#  - get_model selects one of these based on its name. also "register_algorithm" for tracking them
# 2. argument parsing from .yaml file (should implement)
# 3. dataset loading as implemented in src.data.utils.load_dataset (can probably use opso version instead)
#  - this might include an augmentation routine, or that might be in the "model" ie architecture
#  - some augmentation is coming from src.data.spec_augment, we have these in opso.torch.spec_augment I think
# 4. some other utilities and base classes

# copying from https://github.com/zhmiao/BirdMultiLabel/blob/master/src/algorithms/plain_resnet.py

import os
import numpy as np
from datetime import datetime

# from tqdm import tqdm
import random

import torch
import torch.optim as optim
from sklearn.metrics import jaccard_score, hamming_loss, precision_recall_fscore_support

from opensoundscape.torch.architectures.distreg_resnet_architecture import (
    DistRegResNetClassifier,
)
from opensoundscape.torch.models.utils import BaseModule

# from .utils import register_algorithm, Algorithm, single_acc, WarmupScheduler
# from src.data.utils import load_dataset
# from src.models.utils import get_model
#
# from src.data.spec_augment import time_warp, time_mask, freq_mask

import numpy as np


# def load_data(args):
#
#     """
#     Dataloading function. This function can change alg by alg as well.
#     """
#
#     trainloader = load_dataset(
#         name=args.dataset_name,
#         dset="train",
#         num_channels=args.num_channels,
#         rootdir=args.dataset_root,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         augment=args.augment,
#         cas_sampler=True,
#     )
#
#     testloader = load_dataset(
#         name=args.dataset_name,
#         dset="test",
#         num_channels=args.num_channels,
#         rootdir=args.dataset_root,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#     )
#
#     valloader = load_dataset(
#         name=args.dataset_name,
#         dset="val",
#         num_channels=args.num_channels,
#         rootdir=args.dataset_root,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#     )
#
#     return trainloader, testloader, valloader


# NOTE: Turning off all logging for now. may want to use logging module in future

# @register_algorithm("PlainResNet")
# class PlainResNet(Algorithm):
class PlainResNet(BaseModule):

    """
    Overall training function.
    """

    name = "PlainResNet"
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, train_dataset, valid_dataset):
        """if you want to change other parameters,
        simply create the object then modify them
        """
        super(PlainResNet, self).__init__()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # self.num_classes = len(train_dataset.labels)
        self.classes = train_dataset.labels
        print(f"n classes: {len(self.classes)}")

        self.weights_init = "ImageNet"
        self.prediction_threshold = 0.25
        self.num_layers = 18
        # optimization parameters
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

        print(f"Building DistRegResNetClassifier architecture")
        self.network = DistRegResNetClassifier(  # or pass architecture as argument
            # name='PlainResNet',
            num_cls=len(self.classes),
            weights_init=self.weights_init,
            num_layers=self.num_layers,
            init_feat_only=True,
            class_freq=np.array(self.train_class_counts),
        )

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
        net_optim_params_list = [
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
        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt_net, step_size=self.step_size, gamma=self.gamma
        )

        # set up train_loader and valid_loader dataloaders
        # make a dataloader to supply training images from train_dataset
        # eventually should use models.utils.get_dataloader() for cas sampler
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # make a dataloader to supply training images from valid_dataset
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
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

        # TODO : add noise augmentation
        # noise = torch.empty_like(data_2).normal_(mean=0, std=1.0).cuda()
        # data_2 += noise
        # TODO:  labels should include all classes in overlayed imgs

        self.network.train()

        # N = len(self.trainloader)
        # # N = self.total_steps * 3
        # # N = self.total_steps
        #
        # tr_iter = iter(self.trainloader)
        # tr_iter_2 = iter(self.trainloader)
        # tr_iter_3 = iter(self.trainloader)

        for batch_idx, item in enumerate(self.train_loader):
            # all augmentation occurs in the Preprocessor
            data, labels = item["X"], item["y"]
            data = data.to(self.device)
            labels = labels.to(self.device)
            labels = labels.squeeze(1)

            # data, labels = next(tr_iter)
            #
            # # log basic adda train info
            N = len(self.train_loader)
            info_str = "[Train PlainResNet] Epoch: {} [batch {}/{} ({:.2f}%)] ".format(
                epoch, batch_idx, N, 100 * batch_idx / N
            )
            #
            # ########################
            # # Setup data variables #
            # ########################
            # data, labels = data.cuda(), labels.cuda()
            # data.requires_grad = False
            # labels.requires_grad = False
            #
            # with torch.set_grad_enabled(False):
            #     if self.args.augment != 0:
            #         info_str += "-aug- "
            #         data = time_warp(data.clone(), W=self.args.time_warp_W)
            #         data = time_mask(
            #             data, T=self.args.time_mask_T, max_masks=self.args.max_time_mask
            #         )
            #         data = freq_mask(
            #             data, F=self.args.freq_mask_F, max_masks=self.args.max_freq_mask
            #         )
            #     data = torch.cat([data] * 3, dim=1)
            #
            #     noise = torch.empty_like(data).normal_(mean=0, std=1.0).cuda()
            #     data += noise
            #
            #     # -----------
            #     # overlap_switch = random.choice([0, 1, 1])
            #     # # overlap_switch = random.choice([1, 1])
            #     # if overlap_switch == 1:
            #
            #     overlap_switch = torch.tensor(
            #         [random.choice([0, 0, 0, 1]) for _ in range(len(data))]
            #     ).cuda()
            #
            #     data_2, labels_2 = next(tr_iter_2)
            #     data_2, labels_2 = data_2.cuda(), labels_2.cuda()
            #     data_2.requires_grad = False
            #     labels_2.requires_grad = False
            #
            #     if self.args.augment != 0:
            #         data_2 = time_warp(data_2.clone(), W=self.args.time_warp_W)
            #         data_2 = time_mask(
            #             data_2,
            #             T=self.args.time_mask_T,
            #             max_masks=self.args.max_time_mask,
            #         )
            #         data_2 = freq_mask(
            #             data_2,
            #             F=self.args.freq_mask_F,
            #             max_masks=self.args.max_freq_mask,
            #         )
            #     data_2 = torch.cat([data_2] * 3, dim=1)
            #
            #     noise = torch.empty_like(data_2).normal_(mean=0, std=1.0).cuda()
            #     data_2 += noise
            #
            #     overlap_weight = random.randint(5, 10) / 10
            #     data = data + (
            #         data_2 * overlap_weight * overlap_switch.reshape((-1, 1, 1, 1))
            #     )
            #
            #     norm = (
            #         torch.tensor([overlap_weight for _ in range(len(data))])
            #         .cuda()
            #         .reshape((-1, 1, 1, 1))
            #     )
            #     #this is just normalizing after adding images to eachother
            #     data = data / (1.0 + overlap_switch.reshape((-1, 1, 1, 1)) * norm)
            #
            #     #Note: labels are updated to include all classes in overlayed img
            #     labels += labels_2 * overlap_switch.reshape((-1, 1))
            #     labels[labels > 1.0] = 1.0
            #
            #     # overlap_switch = random.choice([0, 0, 0, 1])
            #     # if overlap_switch == 1:
            #     overlap_switch = torch.tensor(
            #         [random.choice([0, 0, 0, 1]) for _ in range(len(data))]
            #     ).cuda()
            #
            #     data_3, labels_3 = next(tr_iter_3)
            #     data_3, labels_3 = data_3.cuda(), labels_3.cuda()
            #     data_3.requires_grad = False
            #     labels_3.requires_grad = False
            #
            #     if self.args.augment != 0:
            #         data_3 = time_warp(data_3.clone(), W=self.args.time_warp_W)
            #         data_3 = time_mask(
            #             data_3,
            #             T=self.args.time_mask_T,
            #             max_masks=self.args.max_time_mask,
            #         )
            #         data_3 = freq_mask(
            #             data_3,
            #             F=self.args.freq_mask_F,
            #             max_masks=self.args.max_freq_mask,
            #         )
            #     data_3 = torch.cat([data_3] * 3, dim=1)
            #
            #     noise = torch.empty_like(data_3).normal_(mean=0, std=1.0).cuda()
            #     data_3 += noise
            #
            #     overlap_weight = random.randint(5, 10) / 10
            #     data = data + (
            #         data_3 * overlap_weight * overlap_switch.reshape((-1, 1, 1, 1))
            #     )
            #     norm = (
            #         torch.tensor([overlap_weight for _ in range(len(data))])
            #         .cuda()
            #         .reshape((-1, 1, 1, 1))
            #     )
            #     data = data / (1.0 + overlap_switch.reshape((-1, 1, 1, 1)) * norm)
            #     # data /= (1 + overlap_weight)
            #
            #     labels += labels_3 * overlap_switch.reshape((-1, 1))
            #     labels[labels > 1.0] = 1.0
            #     # -----------

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
            self.opt_net.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.opt_net.step()

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

    def train(self, num_epochs, log_interval=10, batch_size=1, num_workers=1):

        self.num_epochs = num_epochs
        self.log_interval = log_interval

        self.set_train(batch_size, num_workers)

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(self.num_epochs):

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
        self.save_model()

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

    # "deploy" means test performance with binary predictions
    # def deploy_epoch(self, loader, target_id, target_spp):
    #
    #     self.network.eval()
    #
    #     total_preds = []
    #     total_tgts = []
    #     total_logits = []
    #
    #     # Forward and record # correct predictions of each class
    #     with torch.set_grad_enabled(False):
    #
    #         for data, labels in tqdm(loader, total=len(loader)):
    #
    #             # setup data
    #             data, labels = data.cuda(), labels.cuda()
    #             data.requires_grad = False
    #             labels.requires_grad = False
    #
    #             data = torch.cat([data] * 3, dim=1)
    #
    #             # forward
    #             feats = self.network.feature(data)
    #             logits = self.network.classifier(feats)
    #
    #             total_logits.append(logits.detach().cpu().numpy()[:, target_id])
    #
    #             # Threashold prediction
    #             preds = (
    #                 (torch.sigmoid(logits) >= self.prediction_threshold).int().detach().cpu().numpy()
    #             )
    #             # preds = (torch.sigmoid(logits) >= 0.1).int().detach().cpu().numpy()
    #
    #             preds = preds[:, target_id]
    #
    #             tgts = labels.int().detach().cpu().numpy()
    #
    #             total_preds.append(preds)
    #             total_tgts.append(tgts)
    #
    #     total_logits = np.concatenate(total_logits, axis=0)
    #     total_preds = np.concatenate(total_preds, axis=0)
    #     total_tgts = np.concatenate(total_tgts, axis=0)
    #
    #     # Record per class precision, recall, and f1
    #     class_pre, class_rec, class_f1, _ = precision_recall_fscore_support(
    #         total_tgts, total_preds, average=None, zero_division=0
    #     )
    #
    #     eval_info = "{} Per-class evaluation results: \n".format(
    #         datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #     )
    #     for i in range(len(class_pre)):
    #         eval_info += (
    #             "[Class {}] ".format(i)
    #             + "Pre: {:.3f}; ".format(class_pre[i] * 100)
    #             + "Rec: {:.3f}; ".format(class_rec[i] * 100)
    #             + "F1: {:.3f};\n".format(class_f1[i] * 100)
    #         )
    #
    #     eval_info += (
    #         "Macro Pre: {:.3f}; ".format(class_pre.mean() * 100)
    #         + "Macro Rec: {:.3f}; ".format(class_rec.mean() * 100)
    #         + "Macro F1: {:.3f}\n".format(class_f1.mean() * 100)
    #     )
    #
    #     logits_path = self.weights_path.replace(".pth", "_{}.npz".format(target_spp))
    #     self.main_logger.info("Saving logits and targets to {}".format(logits_path))
    #     np.savez(logits_path, logits=total_logits, targets=total_tgts)
    #
    #     return eval_info

    # def deploy(self, target_spp):
    #
    #     self.set_eval()
    #
    #     #?? maybe the class label?
    #     target_id = self.trainloader.dataset.species_ids[target_spp]
    #
    #     deployloader = load_dataset(
    #         name="POWD",
    #         dset="test",
    #         num_channels=self.args.num_channels,
    #         rootdir=self.args.dataset_root,
    #         batch_size=self.args.batch_size,
    #         shuffle=False,
    #         num_workers=self.args.num_workers,
    #         target_spp=target_spp,
    #     )
    #
    #     eval_info = self.deploy_epoch(
    #         deployloader, target_id=target_id, target_spp=target_spp
    #     )
    #     self.main_logger.info(eval_info)

    def save_model(self):
        os.makedirs(self.weights_path.rsplit("/", 1)[0], exist_ok=True)
        print("Saving to {}".format(self.weights_path))
        self.network.save(self.weights_path)

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

            for sample in loader:
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
        pred_df = pd.DataFrame(
            index=img_paths, data=all_predictions, columns=self.classes
        )

        return pred_df
