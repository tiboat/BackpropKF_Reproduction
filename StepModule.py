from torch import nn
import torch.nn.functional as F
import torch

# Based on: ImageClassificationBase from https://jovian.ai/aakashns/05-cifar10-cnn


class StepModule(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        zt, lt_hat = self(images)  # Generate predictions
        loss = F.mse_loss(zt, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        zt, lt_hat = self(images)  # Generate predictions
        loss = F.mse_loss(zt, labels)  # Calculate loss
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end_no_lr(self, epoch, result):
        print("{},{:.4f},{:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))

    def epoch_end(self, epoch, result):
        print("{},{:.4f},{:.4f},{:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['lr'][-1]))

