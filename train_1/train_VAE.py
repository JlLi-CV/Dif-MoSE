import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from model.autoencoder.autoencoder import AutoencoderKL

import argparse
import matplotlib.pyplot as plt
from Metrics import calc_psnr




class VAETrainer:
    def __init__(self, config):

        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")


        self.ddconfig = config["ddconfig"]
        self.embed_dim = config["embed_dim"]
        self.model = AutoencoderKL(self.ddconfig, self.embed_dim).to(self.device)


        self.start_epoch = 0
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]



        self.data_dir = config["data_dir"]
        self.image_size = config["ddconfig"]["resolution"]
        self.in_channels = config["ddconfig"]["in_channels"]
        self.dataroot = config['dataroot']
        self.dataset = config['dataset']

        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.save_dir, "reconstructions"), exist_ok=True)


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"],
            eta_min=1e-6
        )

        self.train_loader, self.val_loader = self._prepare_dataloaders()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def _prepare_dataloaders(self):

        train_dataset = ''   ## implement a custom data loader
        val_dataset = ''     ## implement a custom data loader

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader

    def _vae_loss(self, recon_x, x, posterior):
        recon_loss = nn.functional.l1_loss(recon_x, x, reduction="mean")
        kl_loss = posterior.kl().mean()
        total_loss = recon_loss +  1e-6 * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_var = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} (Train)")
        for _ , _, data in pbar:
            data = data.to(self.device)


            self.optimizer.zero_grad()
            recon_batch, posterior = self.model(data)
            loss, recon_loss, kl_loss = self._vae_loss(recon_batch, data, posterior)


            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            batch_var = posterior.var.mean().item()
            total_var += batch_var

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon(L1)": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}"
            })


        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon_loss / len(self.train_loader)
        avg_kl = total_kl_loss / len(self.train_loader)
        avg_var = total_var / len(self.train_loader)
        self.train_losses.append(avg_loss)
        print(f"\nEpoch {epoch + 1} train：t_loss={avg_loss:.4f}, L1_loss={avg_recon:.4f}, KL_loss={avg_kl:.4f} , var_mean={avg_var : .6f}")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.epochs} (Val)")
            for batch_idx, (_ , _, data) in enumerate(pbar):
                data = data.to(self.device)

                recon_data, posterior = self.model(data, sample_posterior=False)
                loss, recon_loss, kl_loss = self._vae_loss(recon_data, data, posterior)


                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon(L1)": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_loss.item():.4f}"
                })

                if  epoch % 19 == 0:

                    gt = data.squeeze().detach().cpu().numpy()
                    re = recon_data.squeeze().detach().cpu().numpy()
                    gt = gt.transpose(1 , 2, 0)
                    re = re.transpose(1 , 2, 0)
                    psnr = calc_psnr(gt , re)
                    print('====================> PSNR : {:.4f}'.format(psnr))
                    plt.subplot(1 , 2, 1)
                    plt.imshow(gt[... , 20] , cmap='gray')
                    plt.subplot(1, 2, 2)
                    plt.imshow(re[..., 20], cmap='gray')
                    plt.show()



        avg_loss = total_loss / len(self.val_loader)
        avg_recon = total_recon_loss / len(self.val_loader)
        avg_kl = total_kl_loss / len(self.val_loader)

        self.val_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} ：t_loss={avg_loss:.4f}, L1_loss={avg_recon:.4f}, KL_loss={avg_kl:.4f}")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, os.path.join(self.save_dir, "best_model.pth"))
            print(f"best_val_loss：{self.best_val_loss:.4f}")
        if (epoch + 1) % 100 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, os.path.join(self.save_dir, "epoch_{}.pth").format(epoch + 1))
        return avg_loss



    def run(self):

        print("=" * 50)
        print("training_start")
        print("=" * 50)

        for epoch in range(self.start_epoch , self.epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)
            self.scheduler.step()



if __name__ == "__main__":
 
    train_config = {
        "device": "cuda:0",
        "epochs": 3500,
        "batch_size": 8,
        "lr": 1e-4,
        "data_dir": "./data",
        "save_dir": "../checkpoints/PaviaU/VAE_model/",
        'dataroot':'../../DATA/PaviaU/',
        'dataset':'PaviaU',
        "ddconfig": {
            "ch": 64,
            "ch_mult": (2, 2, 2, 2),
            "num_res_blocks": 2,
            "attn_resolutions": (16, 8),
            "dropout": 0.1,
            "resamp_with_conv": True,
            "in_channels": 103,
            "resolution": 64,
            "z_channels": 64,
            "double_z": True,
            "out_ch": 103,
            "tanh_out": False,
            "use_linear_attn": False,
            "attn_type": "vanilla"
        },
        "embed_dim": 64
    }

    trainer = VAETrainer(train_config)
    trainer.run()