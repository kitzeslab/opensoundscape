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
from tqdm import tqdm
import random

import torch
import torch.optim as optim
from sklearn.metrics import jaccard_score, hamming_loss, precision_recall_fscore_support

from .utils import register_algorithm, Algorithm, single_acc, WarmupScheduler
from src.data.utils import load_dataset
from src.models.utils import get_model

from src.data.spec_augment import time_warp, time_mask, freq_mask

import numpy as np


def load_data(args):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    trainloader = load_dataset(
        name=args.dataset_name,
        dset="train",
        num_channels=args.num_channels,
        rootdir=args.dataset_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augment=args.augment,
        cas_sampler=True,
    )

    testloader = load_dataset(
        name=args.dataset_name,
        dset="test",
        num_channels=args.num_channels,
        rootdir=args.dataset_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    valloader = load_dataset(
        name=args.dataset_name,
        dset="val",
        num_channels=args.num_channels,
        rootdir=args.dataset_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return trainloader, testloader, valloader


@register_algorithm("PlainResNet")
class PlainResNet(Algorithm):

    """
    Overall training function.
    """

    name = "PlainResNet"
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(PlainResNet, self).__init__(args=args)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.testloader, self.valloader = load_data(args)
        _, self.train_class_counts = self.trainloader.dataset.class_counts_cal()

        self.total_steps = int(self.train_class_counts.sum() / args.batch_size)

        self.theta = args.theta

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.main_logger.info("\nGetting {} model.".format(self.args.model_name))
        self.net = get_model(
            name=self.args.model_name,
            num_cls=self.args.num_classes,
            weights_init=self.args.weights_init,
            num_layers=self.args.num_layers,
            init_feat_only=True,
            class_freq=self.train_class_counts,
        )

        ######################
        # Optimization setup #
        ######################
        # Setup optimizer parameters for each network component
        net_optim_params_list = [
            {
                "params": self.net.feature.parameters(),
                "lr": self.args.lr_feature,
                "momentum": self.args.momentum_feature,
                "weight_decay": self.args.weight_decay_feature,
            },
            {
                "params": self.net.classifier.parameters(),
                "lr": self.args.lr_classifier,
                "momentum": self.args.momentum_classifier,
                "weight_decay": self.args.weight_decay_classifier,
            },
        ]
        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt_net, step_size=self.args.step_size, gamma=self.args.gamma
        )

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.main_logger.info("\nGetting {} model.".format(self.args.model_name))
        self.main_logger.info("\nLoading from {}".format(self.weights_path))
        self.net = get_model(
            name=self.args.model_name,
            num_cls=self.args.num_classes,
            weights_init=self.weights_path,
            num_layers=self.args.num_layers,
            init_feat_only=False,
            class_freq=self.train_class_counts,
        )

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)
        # N = self.total_steps * 3
        # N = self.total_steps

        tr_iter = iter(self.trainloader)
        tr_iter_2 = iter(self.trainloader)
        tr_iter_3 = iter(self.trainloader)

        for batch_idx in range(N):

            data, labels = next(tr_iter)

            # log basic adda train info
            info_str = "[Train {}] Epoch: {} [batch {}/{} ({:.2f}%)] ".format(
                self.name, epoch, batch_idx, N, 100 * batch_idx / N
            )

            ########################
            # Setup data variables #
            ########################
            data, labels = data.cuda(), labels.cuda()
            data.requires_grad = False
            labels.requires_grad = False

            with torch.set_grad_enabled(False):
                if self.args.augment != 0:
                    info_str += "-aug- "
                    data = time_warp(data.clone(), W=self.args.time_warp_W)
                    data = time_mask(
                        data, T=self.args.time_mask_T, max_masks=self.args.max_time_mask
                    )
                    data = freq_mask(
                        data, F=self.args.freq_mask_F, max_masks=self.args.max_freq_mask
                    )
                data = torch.cat([data] * 3, dim=1)

                noise = torch.empty_like(data).normal_(mean=0, std=1.0).cuda()
                data += noise

                # -----------
                # overlap_switch = random.choice([0, 1, 1])
                # # overlap_switch = random.choice([1, 1])
                # if overlap_switch == 1:

                overlap_switch = torch.tensor(
                    [random.choice([0, 0, 0, 1]) for _ in range(len(data))]
                ).cuda()

                data_2, labels_2 = next(tr_iter_2)
                data_2, labels_2 = data_2.cuda(), labels_2.cuda()
                data_2.requires_grad = False
                labels_2.requires_grad = False

                if self.args.augment != 0:
                    data_2 = time_warp(data_2.clone(), W=self.args.time_warp_W)
                    data_2 = time_mask(
                        data_2,
                        T=self.args.time_mask_T,
                        max_masks=self.args.max_time_mask,
                    )
                    data_2 = freq_mask(
                        data_2,
                        F=self.args.freq_mask_F,
                        max_masks=self.args.max_freq_mask,
                    )
                data_2 = torch.cat([data_2] * 3, dim=1)

                noise = torch.empty_like(data_2).normal_(mean=0, std=1.0).cuda()
                data_2 += noise

                overlap_weight = random.randint(5, 10) / 10
                data = data + (
                    data_2 * overlap_weight * overlap_switch.reshape((-1, 1, 1, 1))
                )

                norm = (
                    torch.tensor([overlap_weight for _ in range(len(data))])
                    .cuda()
                    .reshape((-1, 1, 1, 1))
                )
                data = data / (1.0 + overlap_switch.reshape((-1, 1, 1, 1)) * norm)

                labels += labels_2 * overlap_switch.reshape((-1, 1))
                labels[labels > 1.0] = 1.0

                # overlap_switch = random.choice([0, 0, 0, 1])
                # if overlap_switch == 1:
                overlap_switch = torch.tensor(
                    [random.choice([0, 0, 0, 1]) for _ in range(len(data))]
                ).cuda()

                data_3, labels_3 = next(tr_iter_3)
                data_3, labels_3 = data_3.cuda(), labels_3.cuda()
                data_3.requires_grad = False
                labels_3.requires_grad = False

                if self.args.augment != 0:
                    data_3 = time_warp(data_3.clone(), W=self.args.time_warp_W)
                    data_3 = time_mask(
                        data_3,
                        T=self.args.time_mask_T,
                        max_masks=self.args.max_time_mask,
                    )
                    data_3 = freq_mask(
                        data_3,
                        F=self.args.freq_mask_F,
                        max_masks=self.args.max_freq_mask,
                    )
                data_3 = torch.cat([data_3] * 3, dim=1)

                noise = torch.empty_like(data_3).normal_(mean=0, std=1.0).cuda()
                data_3 += noise

                overlap_weight = random.randint(5, 10) / 10
                data = data + (
                    data_3 * overlap_weight * overlap_switch.reshape((-1, 1, 1, 1))
                )
                norm = (
                    torch.tensor([overlap_weight for _ in range(len(data))])
                    .cuda()
                    .reshape((-1, 1, 1, 1))
                )
                data = data / (1.0 + overlap_switch.reshape((-1, 1, 1, 1)) * norm)
                # data /= (1 + overlap_weight)

                labels += labels_3 * overlap_switch.reshape((-1, 1))
                labels[labels > 1.0] = 1.0
                # -----------

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.net.feature(data)
            logits = self.net.classifier(feats)
            # calculate loss
            loss = self.net.criterion_cls(logits, labels)

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
                    (torch.sigmoid(logits) >= self.theta).int().detach().cpu().numpy()
                )

                # Jaccard score and Hamming loss
                jacc = jaccard_score(tgts, preds, average="macro")
                hamm = hamming_loss(tgts, preds)

                # log update info
                info_str += "Jacc: {:0.3f} Hamm: {:0.3f} DistLoss: {:.3f}".format(
                    jacc, hamm, loss.item()
                )
                self.main_logger.info(info_str)

        self.scheduler.step()

    def train(self):

        self.set_train()

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(self.num_epochs):

            # Training
            self.train_epoch(epoch)

            # Validation
            self.main_logger.info("\nValidation.")
            val_f1 = self.evaluate(self.valloader)  # , test=False)
            if val_f1 > best_f1:
                self.net.update_best()
                best_f1 = val_f1
                best_epoch = epoch

        self.main_logger.info(
            "\nBest Model Appears at Epoch {} with F1 {:.3f}...".format(
                best_epoch, best_f1 * 100
            )
        )
        self.save_model()

    def evaluate_epoch(self, loader):

        self.net.eval()

        # Get unique classes in the loader and corresponding counts
        loader_uni_class, eval_class_counts = loader.dataset.class_counts_cal()
        total_preds = []
        total_tgts = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, labels in tqdm(loader, total=len(loader)):

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                data = torch.cat([data] * 3, dim=1)

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)

                # Threashold prediction
                preds = (
                    (torch.sigmoid(logits) >= self.theta).int().detach().cpu().numpy()
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
                    i, self.train_class_counts[loader_uni_class][i]
                )
                + "Pre: {:.3f}; ".format(class_pre[i] * 100)
                + "Rec: {:.3f}; ".format(class_rec[i] * 100)
                + "F1: {:.3f};\n".format(class_f1[i] * 100)
            )

        eval_info += (
            "Macro Pre: {:.3f}; ".format(class_pre.mean() * 100)
            + "Macro Rec: {:.3f}; ".format(class_rec.mean() * 100)
            + "Macro F1: {:.3f}\n".format(class_f1.mean() * 100)
        )

        return eval_info, class_f1.mean()

    def evaluate(self, loader):

        if loader == self.testloader:
            self.set_eval()

        eval_info, eval_f1 = self.evaluate_epoch(loader)
        self.main_logger.info(eval_info)

        return eval_f1

    def deploy_epoch(self, loader, target_id, target_spp):

        self.net.eval()

        total_preds = []
        total_tgts = []
        total_logits = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, labels in tqdm(loader, total=len(loader)):

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                data = torch.cat([data] * 3, dim=1)

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)

                total_logits.append(logits.detach().cpu().numpy()[:, target_id])

                # Threashold prediction
                preds = (
                    (torch.sigmoid(logits) >= self.theta).int().detach().cpu().numpy()
                )
                # preds = (torch.sigmoid(logits) >= 0.1).int().detach().cpu().numpy()

                preds = preds[:, target_id]

                tgts = labels.int().detach().cpu().numpy()

                total_preds.append(preds)
                total_tgts.append(tgts)

        total_logits = np.concatenate(total_logits, axis=0)
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
                "[Class {}] ".format(i)
                + "Pre: {:.3f}; ".format(class_pre[i] * 100)
                + "Rec: {:.3f}; ".format(class_rec[i] * 100)
                + "F1: {:.3f};\n".format(class_f1[i] * 100)
            )

        eval_info += (
            "Macro Pre: {:.3f}; ".format(class_pre.mean() * 100)
            + "Macro Rec: {:.3f}; ".format(class_rec.mean() * 100)
            + "Macro F1: {:.3f}\n".format(class_f1.mean() * 100)
        )

        logits_path = self.weights_path.replace(".pth", "_{}.npz".format(target_spp))
        self.main_logger.info("Saving logits and targets to {}".format(logits_path))
        np.savez(logits_path, logits=total_logits, targets=total_tgts)

        return eval_info

    def deploy(self, target_spp):

        self.set_eval()

        target_id = self.trainloader.dataset.species_ids[target_spp]

        deployloader = load_dataset(
            name="POWD",
            dset="test",
            num_channels=self.args.num_channels,
            rootdir=self.args.dataset_root,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            target_spp=target_spp,
        )

        eval_info = self.deploy_epoch(
            deployloader, target_id=target_id, target_spp=target_spp
        )
        self.main_logger.info(eval_info)

    def save_model(self):
        os.makedirs(self.weights_path.rsplit("/", 1)[0], exist_ok=True)
        self.main_logger.info("Saving to {}".format(self.weights_path))
        self.net.save(self.weights_path)
