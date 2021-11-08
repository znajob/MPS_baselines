import argparse
import os
import numpy as np
import torch
from torchvision import transforms, datasets
import pandas as pd

from qmltn.utils.augmentation import image_transformations
from qmltn.torchmps.embeddings import image_embedding


class Evaluator():
    def __init__(self):
        kwargs = vars(self.arg_parser().parse_args())
        self.dataset = kwargs["dataset"]
        self.datadir = kwargs["datadir"]
        self.modelsdir = kwargs["models_dir"]
        self.bs = kwargs["bs"]
        self.device = kwargs["device"]

    def get_files(self):
        files = [file for file in sorted(os.listdir(
            self.modelsdir)) if file.startswith(self.dataset)]
        files.reverse()
        return files

    def get_test_loader(self, crop, bs):
        transform_test = transforms.Compose([
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
        ])

        if self.dataset == "FashionMNIST":
            test_set = datasets.FashionMNIST(self.datadir, download=True, transform=transform_test,
                                             train=False)
        elif self.dataset == "MNIST":
            test_set = datasets.MNIST(self.datadir, download=True, transform=transform_test,
                                      train=False)
        elif self.dataset == "CIFAR10":
            test_set = datasets.CIFAR10(self.datadir, download=True, transform=transform_test,
                                        train=False)
        else:
            raise Exception("Dataset not available.")

        ntest = len(test_set)
        itest = range(ntest)
        test_sampler = torch.utils.data.SequentialSampler(itest)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs,
                                                  sampler=test_sampler, drop_last=True)

        return test_loader, len(test_set)//bs

    def evaluate(self, model_name, bs):
        model = torch.load(os.path.join(self.modelsdir, model_name),
                           map_location=torch.device(self.device))
        crop = int(model_name.split("s")[-1].split("f")[0].split("c")[1])
        test_loader, num_batches = self.get_test_loader(crop, bs)
        # Evaluate accuracy of MPS classifier on the test set
        running_acc = 0.
        out = []
        out_scores = []
        with torch.no_grad():
            k = 0
            for inputs, labels in test_loader:
                k += 1
                labels = labels.data

                inputs = image_embedding(inputs, aug_phi=0)

                if model.permute is not None:
                    inputs = inputs[:, model.permute, :]

                # Call our MPS to get logit scores and predictions
                if self.device == "cuda":
                    inputs = inputs.to(self.device)
                    scores = model(inputs).cpu()
                else:
                    scores = model(inputs)

                out_scores += list(scores.numpy())
                _, preds = torch.max(scores, 1)
                out += list(preds)
                acc = torch.sum(preds == labels).item() / len(labels)
                running_acc += acc
                print(
                    f"\r{model_name}, batch: {k}/{num_batches}, acc={acc:.3f}", end="")
            print(
                f"\r{model_name} batch={k}/{len(test_loader)} acc={running_acc/num_batches:.4f}")

        return out, out_scores

    def get_model_scores(self):
        bs = self.bs
        model_scores = {}

        test_loader, _ = self.get_test_loader(20, bs)
        model_scores["labels"] = []
        for inputs, labels in test_loader:
            model_scores["labels"] += list(labels.numpy())

        model_names = self.get_files()
        print(f"Evaluating {len(model_names)} models.")
        with torch.no_grad():
            for i, model_name in enumerate(model_names):
                if model_name not in model_scores.keys():
                    try:
                        _, scores = self.evaluate(model_name, bs)
                        model_scores[model_name] = list(scores)
                    except:
                        print(f"Failed to evaluate {model_name}")
                        if model_name in model_scores.keys():
                            model_scores.pop(model_name)
        print("Constructing the dataframe")
        df_scores = pd.DataFrame.from_dict(model_scores)
        print("Saving the dataframe")
        df_scores.to_csv(os.path.join(
            self.modelsdir, f"evaluation_{self.dataset}.csv"), index=False)

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset',
                            default="MNIST",
                            type=str,
                            help='The name of the dataset.')
        parser.add_argument('--datadir',
                            default="../dataset",
                            type=str,
                            help='A full to datasets.')
        parser.add_argument('--models_dir',
                            type=str,
                            help='Directory where models to evaluate are stored.')
        parser.add_argument('--bs',
                            default="100",
                            type=int,
                            help='Batch size for evaluation.')
        parser.add_argument('--device',
                            type=str,
                            default="cuda",
                            help='Device used to compute the scores. cuda (default), cpu')

        return parser


if __name__ == '__main__':
    e = Evaluator()
    e.get_model_scores()
