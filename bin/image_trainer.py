from qmltn.torchmps.trainer.trainer_cli import ImageTrainerCLI
from qmltn.torchmps.torchmps import build_mps_model


class ImageTrainer(ImageTrainerCLI):
    def __init__(self):
        super(ImageTrainer, self).__init__()

    def build_model(self, *args, **kwargs):
        self.model, config = build_mps_model(**self.config)
        self.epoch = config["epoch"]
        self.best_val_acc = config["best_val_acc"]
        self.best_val_acc_epoch = config["best_val_acc_epoch"]
        self.num_model_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        self.config["num_model_params"] = self.num_model_params
        if hasattr(self.model, "wandb_name"):
            # Determine the previous wandb model name to continue logging to the same run
            self.wandb_name = self.model.wandb_name
        if hasattr(self.model, "wandb_id"):
            # Determine the previous wandb model id to continue logging to the same run
            self.wandb_id = self.model.wandb_id

    def get_wandb_project(self, *args, **kwargs):
        return "MPS"

    def get_model_name(self, *args, **kwargs):
        modelname = f"{self.prefix}{self.dataset}_D{self.bond_dim}l{self.learn_rate}s{self.step_size}g{self.gamma}ne{self.num_epochs}s{self.seed}c{self.cropSize}f{self.fold}_{self.nfolds}e{self.embedding_order}r{self.train_set_ratio}l{self.l2_reg}"
        if self.random_crop:
            modelname += "_r_"
        if self.permute:
            modelname += "p"
        if self.spiral:
            modelname += "s"
        if self.periodic_bc or self.ti:
            modelname += "c"
        else:
            modelname += "o"
        if self.ti:
            modelname += "ti"

        return modelname


if __name__ == "__main__":
    trainer = ImageTrainer()
    if trainer.profile_model:
        trainer.profile()
    trainer.train()
