import argparse
import os
import numpy as np
import torch
from torchvision import transforms, datasets
import pandas as pd

from qmltn.utils.augmentation import image_transformations
from qmltn.torchmps.embeddings import image_embedding
from qmltn.torchmps.torchmps import entropy


class MPSEntropy():
    def __init__(self):
        kwargs = vars(self.arg_parser().parse_args())
        self.dataset = kwargs["dataset"]
        self.modelsdir = kwargs["models_dir"]
        self.load_old = kwargs["load_old"]

    def get_files(self):
        files = [file for file in sorted(os.listdir(
            self.modelsdir)) if file.startswith(self.dataset)]
        files.reverse()
        return files

    def entropy(self, model_name):
        model = torch.load(os.path.join(self.modelsdir, model_name),
                           map_location="cpu")
        return entropy(model)

    def get_model_entropies(self):
        model_ent = {}

        filename = os.path.join(self.modelsdir, f"entropies_{self.dataset}.csv")
        model_names = self.get_files()

        load_old = False
        if self.load_old and os.path.exists(filename):
              df_ent_old = pd.read_csv(filename).set_index("name")
              old_names = list(df_ent_old.index)
              model_names = [name for name in model_names if name not in old_names]
              load_old = True

        print(f"Evaluating {len(model_names)} models.")
        with torch.no_grad():
            for i, model_name in enumerate(model_names):
                # if model_name.endswith("ti"):
                #     continue
                if model_name not in model_ent.keys():
                    try:
                        ent = self.entropy(model_name)
                        print(f"{model_name}: {ent:3.4f}")
                        model_ent[model_name] = ent
                    except:
                        print(f"Failed to evaluate {model_name}")
                        if model_name in model_ent.keys():
                            model_ent.pop(model_name)
        print("Constructing the dataframe")
        df_ent = pd.DataFrame.from_dict(
            model_ent, orient="index", columns=["entropy"])
        if load_old:
            df_ent = pd.concat([df_ent,df_ent_old])
        print("Saving the dataframe")
        df_ent.to_csv(filename, index=True, index_label="name")

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset',
                            default="MNIST",
                            type=str,
                            help='The name of the dataset.')
        parser.add_argument('--models_dir',
                            type=str,
                            help='Directory where models to evaluate are stored.')
        parser.add_argument('--load_old',
                            type=int,
                            default=0,
                            help='If enabled we first load the old results if they exist and then compute what is left. 0 (default - disabled), 1 (enabled)')

        return parser


if __name__ == '__main__':
    e = MPSEntropy()
    e.get_model_entropies()
