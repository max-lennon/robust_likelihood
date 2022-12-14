from utils import *
from train_model import *
from model import NormalizingFlow
import loss
import os
import argparse
import warnings
import yaml
import torch
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

class DatasetWrapper():
    def __init__(self, dataset_type, source, train, resize=None, flip=None):
        dataset_library = {"cifar10": datasets.CIFAR10,
        "imagefolder": datasets.ImageFolder}

        self.transforms = [transforms.ToTensor()]
        if resize is not None:
            self.transforms.append(transforms.Resize(resize))
        if flip is not None:
            self.transforms.append(transforms.RandomHorizontalFlip(p=flip))

        self.param_dict = {"root": source, "transform": transforms.Compose(self.transforms)}

        if dataset_type in ["cifar-10"]:
            self.param_dict["train": train]

        self.dataset = dataset_library[dataset_type](**self.param_dict)

    def set_loader_params(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        config_dict = yaml.safe_load(stream)
        cfg.set_config(config_dict)
    
    batch_size = cfg["batch_size"] // 2 if cfg["dual_objective"] else cfg["batch_size"]
    
    num_pixels = (cfg["image_size"] ** 2) * 3

    ring_library = {"ring_95": lambda y, batch_size : loss.ring_prob(y, batch_size, mode="ring_95"), \
        "ring_50": lambda y, batch_size : loss.ring_prob(y, batch_size, mode="ring_50")}

    offset_gauss_library = {"fixed_direction": lambda y, batch_size: loss.offset_gauss_prob(y, batch_size, \
        direction=torch.ones((batch_size, num_pixels)).to(cfg["device"]) / num_pixels, offset=4)} 

    distribution_library = {"ring": ring_library, "offset_gauss": offset_gauss_library}
    optimizer_library = {"adam": torch.optim.Adam}
    scheduler_library = {"step": torch.optim.lr_scheduler.StepLR, "cyclic": torch.optim.lr_scheduler.CyclicLR}

    experiment_dir = os.path.join(cfg["home_dir"], "experiments", cfg["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)

    os.system( f'cp {args.config_path} {experiment_dir}/{cfg["experiment_name"]}.yaml' )

    train_set_in = DatasetWrapper("cifar10", source='~/Documents/Research/Datasets/cifar-10-python/', train=True, \
        resize=cfg["image_size"], flip=cfg["train_p_flip"])
    test_set_in = DatasetWrapper("cifar10", source='~/Documents/Research/Datasets/cifar-10-python/', train=False, \
        resize=cfg["image_size"])

    train_set_out = DatasetWrapper("imagefolder", source='~/Documents/Research/robust_likelihood/ood_data/train/', \
        train=True, resize=cfg["image_size"], flip=cfg["train_p_flip"])
    test_set_out = DatasetWrapper("imagefolder", source='~/Documents/Research/robust_likelihood/ood_data/test/', \
        train=False, resize=cfg["image_size"])

    for dataset in [train_set_in, test_set_in, train_set_out, test_set_out]:
        dataset.set_loader_params(batch_size, cfg["data_shuffle"])

    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "runs"))

    model = NormalizingFlow(**cfg["norm_flow_parameters"], shape=(3, cfg["image_size"], \
        cfg["image_size"])).to(cfg["device"])
    optimizer = optimizer_library[cfg["optimizer"]](model.parameters(), lr=cfg["learning_rate"])
    scheduler = scheduler_library[cfg["scheduler"]](optimizer, **cfg["scheduler_parameters"])

    checkpoint_dir = os.path.join(experiment_dir, 'ckpts')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, 'ckpts', cfg["checkpoint"])

    epoch = cfg["start_epoch"]

    if cfg["load_checkpoint"]:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=cfg["device"])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(cfg["device"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            best_validation = loss
        except FileNotFoundError:
            warnings.warn(f"Could not find the specified checkpoint. Starting from epoch {cfg['start_epoch']} \
                as specified by config file.")
            best_validation = None
    else:
        best_validation = None

    loss_fn = loss.loss_fn
    ood_loss_fn = lambda y, s, norms, scale, batch_size: loss.loss_fn_ood(\
        distribution_library[cfg["ood_dist"]][cfg["ood_mode"]], y, s, norms, scale, batch_size)

    for t in range(epoch, cfg["epochs"]):
        print(f"Epoch {t+1}\n-------------------------------")
        if cfg["dual_objective"]:
            train_loss = dual_train_loop(train_set_in.get_loader(), train_set_out.get_loader(), model, loss_fn, \
                ood_loss_fn, optimizer, batch_size, t, writer, report_iters=cfg["report_iters"], \
                    num_pixels=num_pixels)

            validation_loss = dual_test_loop(test_set_in.get_loader(), test_set_out.get_loader(), model, loss_fn, \
                ood_loss_fn, t, writer, num_pixels=num_pixels)
        else:
            train_loss = train_loop(train_set_in.get_loader(), model, loss_fn, optimizer, batch_size, t, \
                writer, report_iters=cfg["report_iters"], num_pixels=num_pixels)

            validation_loss = test_loop(test_set_in.get_loader(), model, loss_fn, t, writer, num_pixels=num_pixels)

        if best_validation is None or validation_loss < best_validation:
            best_validation = validation_loss
            torch.save({
                'epoch': t + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validation_loss,
            }, checkpoint_path)
        scheduler.step()