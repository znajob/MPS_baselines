#!/usr/bin/env python3
import os
import time
import signal
import argparse
import subprocess
import numpy as np
from abc import ABC, abstractmethod
import torch
from torchvision import transforms, datasets
import logging
import wandb
from tqdm import tqdm
# from torch.profiler import profile, record_function, ProfilerActivity


from qmltn import __version__
from qmltn.torchmps.torchmps import MPS, TI_MPS
from qmltn.torchmps.embeddings import image_embedding, spiral_inds
from qmltn.utils.killer import GracefulKiller
from qmltn.utils.dataset import init_loaders
from qmltn.torchmps.torchmps import build_mps_model, entropy

class Trainer(torch.nn.Module):
    def __init__(self, killer=GracefulKiller(), *args, **kwargs):
        super(Trainer, self).__init__()

        self.config = kwargs

        # Random crop augmentation handling
        if kwargs["aug_random_crop"] and not kwargs["ti"]:
            logging.warning("The random_crop augmentation can be used only in translational invariant models. Setting random_crop to false")
            self.config["aug_random_crop"] = 0
        self.random_crop = self.config["aug_random_crop"]

        # Image embedding for coloured images
        self.use_grayscale = kwargs["use_grayscale"]

        # Verbosity
        verbose = kwargs["verbose"]
        self.verbose = logging.WARNING
        self.disable_tqdm = True
        if verbose == 1:
            self.verbose = logging.INFO
            self.disable_tqdm = False
        elif verbose == 2:
            self.verbose = logging.DEBUG
            self.disable_tqdm = False
        logging.basicConfig(format='%(asctime)s | %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=self.verbose)

        logging.info(f"Initializing trainer v {__version__}")

        # Folder managment
        self.dataset = kwargs["dataset"]
        self.savedir = os.path.join(kwargs["savedir"], kwargs["dataset"])
        self.datadir = kwargs["datadir"]

        # Miscellaneous initialization
        self.seed = kwargs["seed"]
        self.init_time = time.time()
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.checkpoint = kwargs["checkpoint"]

        # General parameters
        self.cropSize = kwargs["crop"]
        self.output_dim = kwargs["nclass"]
        self.input_dim = self.get_input_dim(**self.config)
        self.config["input_dim"] = self.input_dim
        self.fold = kwargs["fold"]
        self.nfolds = kwargs["nfolds"]

        self.feature_dim = self.get_feature_dim(**self.config)
        self.config["feature_dim"] = self.feature_dim

        # Embedding
        self.permute = kwargs["permute"]
        self.spiral = kwargs["spiral"]
        self.embedding_order = kwargs["embedding_order"]
        self.embedding = kwargs["embedding"]
        if self.spiral and self.permute:
            logging.warning(
                f"Both spiral and permute can not be set to true. Setting permute to false.")
            self.permute = False
        self.inds = None
        if self.permute:
            self.inds = torch.randperm(self.input_dim)
        if self.spiral:
            self.inds = spiral_inds(self.cropSize)
        self.config["inds"] = self.inds

        # MODEL parameters
        self.bond_dim = kwargs["D"]
        self.periodic_bc = kwargs["periodic"]
        self.ti = kwargs["ti"]
        self.aug_phi = kwargs["aug_phi"]

        # Training parameters
        self.train_set_ratio = kwargs["train_ratio"]
        self.adaptive_mode = False
        self.num_train = kwargs["ntrain"]
        self.num_test = kwargs["ntest"]
        self.batch_size = kwargs["bs"]
        self.num_epochs = kwargs["nepoch"]
        self.learn_rate = kwargs["lr"]
        self.l2_reg = kwargs["l2"]
        self.step_size = kwargs["step"]
        self.gamma = kwargs["gamma"]
        self.stop_patience = kwargs["stop_patience"]

        # Device parameters
        self.cuda = kwargs["cuda"]
        if self.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Saving parameters
        self.prefix = kwargs["prefix"]

        self.savemodel = kwargs["savemodel"]
        self.modelname = self.get_model_name(**self.config)

        self.model_path = os.path.join(self.savedir, self.modelname)
        self.config["model_path"] = self.model_path

        # Simulation/training options
        self.max_training_hours = kwargs['max_training_hours']
        self.start_time = time.time()
        if self.max_training_hours > 0:
            signal.alarm(int(3600*self.max_training_hours))
        self.continue_training = kwargs['continue_training']
        self.reset_early_stopping = kwargs['reset_early_stopping']

        self.killer = killer

        # Enabling/disabling test set evaluation
        self.monitoring = kwargs['monitoring']

        # Monitoring using wandb
        self.wandb = kwargs["wandb"]
        self.wandb_offline = kwargs["wandb_offline"]
        if self.wandb:

            if self.wandb_offline:
                os.environ["WANDB_API_KEY"] = "" //Enter your API key here
                os.environ["WANDB_MODE"] = "dryrun"

            self.wandb_project = self.get_wandb_project()
            if kwargs["entity"] == "":
                self.entity = None
            else:
                self.entity = kwargs["entity"]
            self.wandb_name = None
            self.wandb_id = None

        self.monitor_entropy = kwargs["monitor_ent"]

        # Profiling
        self.profile_model = kwargs["profile"]

        # Main objects for training
        self.optimizer_str = kwargs["optimizer"]
        self.model = None
        self.loaders = None
        self.optimizer = None
        self.loss_fun = None
        self.scheduler = None

        self.init_training(*args, **kwargs)

    # Initialize the model
    @abstractmethod
    def build_model(self, *args, **kwargs):
        raise NotImplementedError()

    # Initialize the training and test sets loaders
    @abstractmethod
    def init_loaders(self, *args, **kwargs):
        # The method should output the training and the test datasets including augmentation
        raise NotImplementedError()

    # Get the name for the trained model
    @abstractmethod
    def get_model_name(self, *args, **kwargs):
        raise NotImplementedError()

    # Returns the feature dimension for the model
    @abstractmethod
    def get_feature_dim(self, *args, **kwargs):
        raise NotImplementedError()

    # Returns the input dimension data
    @abstractmethod
    def get_input_dim(self, *args, **kwargs):
        raise NotImplementedError()

    # Returns wandb project name
    @abstractmethod
    def get_wandb_project(self, *args, **kwargs):
        raise NotImplementedError()

    def init_optimizer(self):
        logging.info(f"Initializing the optimizer.")
        # Set our loss function, optimizer, and scheduler
        self.loss_fun = torch.nn.CrossEntropyLoss()
        if self.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate,
                                              weight_decay=self.l2_reg)
        elif self.optimizer_str == "adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learn_rate,
                                                  rho=0.9, eps=1e-06, weight_decay=self.l2_reg)
        elif self.optimizer_str == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learn_rate,
                                                 lr_decay=0, weight_decay=self.l2_reg,
                                                 initial_accumulator_value=0, eps=1e-10)
        elif self.optimizer_str == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate,
                                               betas=(0.9, 0.999), eps=1e-08, weight_decay=self.l2_reg, amsgrad=False)
        elif self.optimizer_str == "adamax":
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learn_rate,
                                                betas=(0.9, 0.999), eps=1e-08, weight_decay=self.l2_reg)
        elif self.optimizer_str == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learn_rate,
                                                 alpha=0.99, eps=1e-08, weight_decay=self.l2_reg, momentum=0, centered=False)
        elif self.optimizer_str == "rprop":
            self.optimizer = torch.optim.Rprop(self.model.parameters(), lr=self.learn_rate,
                                               etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif self.optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learn_rate,
                                             momentum=0, dampening=0, weight_decay=self.l2_reg, nesterov=False)
        else:
            logging.warning(
                "Specified optimizer not found using the default Adam optimizer.")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate,
                                              weight_decay=self.l2_reg)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=self.gamma, patience=self.step_size,
            threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08,
            verbose=self.verbose in [logging.INFO, logging.DEBUG],
            num_bad_epochs=self.epoch-self.best_val_acc_epoch, best_accuracy=self.best_val_acc)

        # We need to update this for graceful killer to have the learning rate even if the trainer has not finished one step.
        self.scheduler._last_lr = [self.learn_rate]

    # Initializes everything necessary for training
    def init_training(self, *args, **kwargs):
        logging.info("Initialize training")

        self.epoch = 0
        self.best_val_acc = 0.0
        self.best_val_acc_epoch = 0

        if self.model is None:
            self.build_model()
        if self.optimizer is None or self.scheduler is None or self.loss_fun is None:
            self.init_optimizer()
        if self.loaders is None:
            self.init_loaders(*args, **kwargs)

        # Preparing the checkpoint directory
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

        logging.info(
            f"Training on {self.dataset} dataset for {self.num_epochs} epochs")
        logging.info(f"Maximum MPS bond dimension = {self.bond_dim}")
        logging.info(
            f" * {'Adaptive' if self.adaptive_mode else 'Fixed'} bond dimensions")
        logging.info(
            f" * {'Periodic' if self.periodic_bc else 'Open'} boundary conditions")
        logging.info(f"Using optimizer: {self.optimizer}")
        if self.l2_reg > 0:
            logging.info(f" * L2 regularization = {self.l2_reg:.2e}")

        if self.permute:
            inds = torch.randperm(self.cropSize**2)

    def train(self):
        # We initialize the wandb at the end in case of training continuation
        run = None
        if self.wandb:
            if (self.wandb_name is not None) and (self.wandb_id is not None):
                logging.info(
                    f"Rsuming wandb job {self.wandb_name}, {self.wandb_id}.")
                run = wandb.init(
                    entity=self.entity, project=f"{self.dataset}_{self.wandb_project}", name=self.wandb_name, config=self.config, resume=True, id=self.wandb_id)
            else:
                run = wandb.init(
                    entity=self.entity, project=f"{self.dataset}_{self.wandb_project}", name=self.wandb_name, config=self.config)
            self.model.wandb_name = run.name
            self.model.wandb_id = run.id
            self.wandb_name = run.name
            self.wandb_id = run.id
            print("wandb.run.id = ", run.id)

        # Let's start training!
        init_epoch = self.epoch
        logging.info("Start training!")
        for epoch_num in range(init_epoch, self.num_epochs+1):
            train_loss, train_acc, val_acc = self.train_step(epoch_num)

            if (self.killer.kill_now or self.stop_patience < epoch_num-self.best_val_acc_epoch):
                logging.warning(
                    f"Early stopping in epoch {epoch_num}. The accuracy did not increase since epoch {self.best_val_acc_epoch}.")
                break

            lr = self.scheduler._last_lr[0]
            wandb_logs = {'loss': train_loss, 'accuracy': train_acc,
                          'validation_accuracy': val_acc, 'epoch': epoch_num, "lr": lr}

            if self.monitoring:
                test_acc = self.evaluate()
                wandb_logs["test_accuracy"] = test_acc

            if self.monitor_entropy:
                ent = entropy(self.model)
                wandb_logs["entropy"] = ent

            if self.wandb:
                run.log(wandb_logs)

        running_time = int(time.time()-self.start_time)
        logging.warning(
            f"Ending training. Running time: {running_time/3600.:.3f} hours, Max training time: {self.max_training_hours} hours.")

        self.update_saved_model()
        # self.run_continue_script()

        # Finalizing wandb
        if self.wandb:
            run.finish()

    # Useful to automatically add a new job to continue training on the cluster
    def run_continue_script(self):
        if self.continue_training:
            run_script = './../scripts/continue_training.sh'
            run_name = self.modelname
            run_string = 'python image_trainer.py'
            run_params = []
            ndigits = 2
            for key, value in list(self.config.items()):
                if isinstance(value, str):
                    sv = f'--{key} {value}'
                else:
                    sv = (
                        f'--{key} ')+'{:g}'.format(float('{:.{p}g}'.format(value, p=ndigits)))
                run_params.append(sv)
            run_string += ' '.join(run_params)
            logging.warning(f"Continuing training for {run_name}")
            subprocess.call(
                f'{run_script} {run_name} "{run_string}"', shell=True)

    def update_saved_model(self):
        if (os.path.exists(self.model_path)):
            saved_model = torch.load(self.model_path)
            lr = self.scheduler._last_lr[0]
            logging.info(f"Last learning rate: {lr:.3e}")

            if hasattr(saved_model, "config"):
                saved_model.config.update({"lr": lr})
            else:
                self.config.update({"lr": lr})
                logging.info("New self.model.config.")
                saved_model.config = self.config

            saved_model.epoch = self.epoch
            saved_model.best_val_acc = self.best_val_acc
            saved_model.best_val_acc_epoch = self.best_val_acc_epoch

            torch.save(saved_model, self.model_path)

            logging.info(
                f"Updated the best saved model {self.modelname} with acc {saved_model.best_val_acc:.3f} in epoch {saved_model.best_val_acc_epoch}. Current epoch: {saved_model.epoch}\n")

    def train_step(self, epoch_num):
        self.model.train()
        running_loss = 0.
        running_acc = 0.
        for inputs, labels in tqdm(self.loaders['train'], disable=self.disable_tqdm):
            if self.killer.kill_now:
                break

            labels = labels.data
            # TODO move the image embedding to the model...

            if self.random_crop:
                crop = self.config["crop"]
                dims = inputs.shape
                size = dims[2]
                ds = size-crop
                sizex,sizey = np.random.randint(2*ds+1,size=2)-ds+crop
                dx = np.random.randint(size-sizex+1)
                dy = np.random.randint(size-sizey+1)
                inputs = inputs[:,:,dx:dx+sizex,dy:dy+sizey]

            inputs = image_embedding(
                inputs, aug_phi=self.aug_phi, embedding=self.embedding, embedding_order=self.embedding_order)

            if self.killer.kill_now:
                break

            # TODO move the inds to the model...
            if self.inds is not None:
                inputs = inputs[:, self.inds, :]

            if self.killer.kill_now:
                break

            # Call our model to get logit scores and predictions
            if self.cuda:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            scores = self.model(inputs)
            _, preds = torch.max(scores, 1)

            # Compute the loss and accuracy, add them to the running totals
            loss = self.loss_fun(scores, labels)
            with torch.no_grad():
                accuracy = torch.sum(preds == labels).item() / self.batch_size
                running_loss += loss
                running_acc += accuracy

            if self.killer.kill_now:
                break

            # Backpropagate and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        val_acc = 0
        train_loss = 0
        train_acc = 0
        if self.killer.running:
            train_loss = running_loss / self.num_batches['train']
            train_acc = running_acc / self.num_batches['train']
            running_time = int(time.time()-self.start_time)
            logging.info(f"### Epoch {self.epoch} ###")
            logging.info(
                f"Average loss:           {train_loss:.4f}")
            logging.info(
                f"Average train accuracy: {train_acc:.4f}")
            logging.info(
                f"Runtime so far:         {running_time} sec")

            # Validating
            val_acc = self.validate()

            # Learning rate scheduler
            self.scheduler.step(val_acc)

            # Increasing the number of trained epochs
            self.epoch += 1

            # Checkpointing the model
            # This should be disabled in the tuner but can be enabled in the trainer
            self.save_best(val_acc, epoch_num)
        return train_loss, train_acc, val_acc

    def save_best(self, val_acc, epoch_num):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_acc_epoch = epoch_num
            if self.savemodel:
                lr = self.scheduler._last_lr[0]
                logging.info(f"Last learning rate: {lr}")

                if hasattr(self.model, "config"):
                    self.model.config.update({"lr": lr})
                else:
                    self.config.update({"lr": lr})
                    logging.info("New self.model.config.")
                    self.model.config = self.config

                self.model.epoch = epoch_num
                self.model.best_val_acc = self.best_val_acc
                self.model.best_val_acc_epoch = self.best_val_acc_epoch

                torch.save(self.model, self.model_path)
                logging.info(
                    f"Saved new best model {self.modelname} with acc {val_acc:.4f} in epoch {epoch_num}.\n")

    def validate(self):
        # Evaluate accuracy of MPS classifier on the val set
        self.model.eval()
        running_acc = 0.
        with torch.no_grad():
            nb = 0
            for inputs, labels in self.loaders['val']:
                if self.killer.kill_now:
                    break
                labels = labels.data

                inputs = image_embedding(
                    inputs, aug_phi=0, embedding=self.embedding, embedding_order=self.embedding_order)

                if self.inds is not None:
                    inputs = inputs[:, self.inds, :]

                if self.cuda:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                # Call our MPS to get logit scores and predictions
                scores = self.model(inputs)
                _, preds = torch.max(scores, 1)
                running_acc += torch.sum(preds ==
                                         labels).item() / len(inputs)
                nb += 1
        logging.info(
            f"Validation accuracy:    {running_acc / nb:.4f}\n")

        return running_acc / nb

    def evaluate(self):
        # Evaluate accuracy of MPS classifier on the test set
        self.model.eval()
        running_acc = 0.
        with torch.no_grad():
            nb = 0
            for inputs, labels in self.loaders['test']:
                labels = labels.data

                inputs = image_embedding(
                    inputs, aug_phi=0, embedding=self.embedding, embedding_order=self.embedding_order)

                if self.inds is not None:
                    inputs = inputs[:, self.inds, :]

                if self.cuda:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                # Call our MPS to get logit scores and predictions
                scores = self.model(inputs)
                _, preds = torch.max(scores, 1)
                running_acc += torch.sum(preds ==
                                         labels).item() / len(inputs)
                nb += 1
        logging.info(
            f"Test accuracy:          {running_acc / nb:.4f}\n")

        return running_acc / nb

    def forward(self, input_data):
        if self.cuda:
            input_data = input_data.to(self.device)
        return self.model.forward(input_data)

    def call(self, input_data):
        if self.cuda:
            input_data = input_data.to(self.device)
        return self.model.forward(input_data)

    def profile(self):
        logging.WARNING(
            "Profiling is not yet implemented. Skiping profiling and continuing.")
        # raise Exception("Not yet implemented")
        # logging.info("Profiling...")
        # inputs = torch.randn(self.batch_size, self.input_dim, self.feature_dim)
        # activities = [ProfilerActivity.CPU]
        # if self.cuda:
        #     activities.append(ProfilerActivity.CUDA)
        #     inputs = inputs.to(self.device)

        # with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        #     with record_function("model_inference"):
        #         self.model(inputs)

        # logging.info(prof.key_averages().table(sort_by="cuda_time_total"))
        # prof.export_chrome_trace(self.model_path+"_trace.json")


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, num_bad_epochs=0, best_accuracy=0):

        self.num_bad_epochs_ = num_bad_epochs
        self.num_bad_epochs = num_bad_epochs
        self.best_accuracy = best_accuracy

        super(ReduceLROnPlateau, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                                                threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr,
                                                eps=eps, verbose=verbose)

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        logging.info(
            f"Reseting the number of bad epochs to {self.num_bad_epochs_}. Best acc = {self.best_accuracy}.")
        self.best = self.best_accuracy
        self.cooldown_counter = 0
        self.num_bad_epochs = self.num_bad_epochs_
