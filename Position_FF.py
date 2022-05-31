import sys
import torch
from matplotlib import pyplot as plt
import Variance
from CircleGenerator import CirclesDataset, create_directory
from torch.utils.data import DataLoader
from DeviceDataLoader import DeviceDataLoader, get_default_device
import torch.nn.functional as F
from Feedforward import FFNetwork
from datetime import datetime


def evaluate(model, val_loader):
    model.eval()
    output_loss_list = []
    pred_pos_list = []
    label_pos_list = []
    for batch in val_loader:
        output_loss, pred_pos, label_pos = validation_step(model, batch)
        output_loss_list.append(output_loss)
        pred_pos_list.append(pred_pos)
        label_pos_list.append(label_pos)
    return model.validation_epoch_end(output_loss_list), pred_pos_list, label_pos_list


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('val_loss') for x in history]
    plt.figure()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def training_step(model, batch):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height), labels: (N, T, (x, y, vx, vy))
    images, labels = batch
    N, T, C, W, H = images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]
    un_sequenced_batch = images.view(N*T, C, W, H)

    positions = labels[:, :, :2]
    un_sequenced_positions = positions.view(N*T, -1)
    zt, l_hats = model(un_sequenced_batch)  # Generate predictions
    loss = F.mse_loss(zt, un_sequenced_positions)  # Calculate loss
    return loss, zt, un_sequenced_positions


def validation_step(model, batch):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height), labels: (N, T, (x, y, vx, vy))
    images, labels = batch
    N, T, C, W, H = images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]
    un_sequenced_batch = images.view(N * T, C, W, H)
    positions = labels[:, :, :2]
    un_sequenced_positions = positions.view(N * T, -1)
    zt, _ = model(un_sequenced_batch)  # Generate predictions
    loss = F.mse_loss(zt, un_sequenced_positions)  # Calculate loss
    return {'val_loss': loss.detach()}, zt, un_sequenced_positions


def fit(epochs, lr, model, train_loader, val_loader, opt_func, lr_scheduling, path_model):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    sched = None
    loss = None
    lowest_val_loss = None
    model_dict = None
    pred_pos_train_list = []
    label_pos_train_list = []
    pred_pos_val_list = []
    label_pos_val_list = []

    # Set up one-cycle learning rate scheduler
    if lr_scheduling:
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss, pred_pos_train, label_pos_train = training_step(model, batch)
            # Save predicted position and labeled position for last epoch
            if epoch == epochs-1:
                pred_pos_train_list.append(pred_pos_train)
                label_pos_train_list.append(label_pos_train)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduling:
                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()
        # Validation phase
        result, pred_pos_val_list, label_pos_val_list = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        if lr_scheduling:
            result['lr'] = lrs
            model.epoch_end(epoch, result)
        else:
            model.epoch_end_no_lr(epoch, result)
        history.append(result)

        if path_model is not None:
            # Save model if lowest val loss
            val_loss = result['val_loss']
            if lowest_val_loss is None or lowest_val_loss > val_loss:
                lowest_val_loss = val_loss
                model_dict = {
                    "epoch": epoch,
                    "ff_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }

    done = datetime.now().strftime("%d-%m-%Y_%H-%M")
    model_dir = f"{path_model}_loss_{lowest_val_loss}_{done}"
    create_directory(model_dir)

    torch.save(model_dict, f"{model_dir}/lowest_val_loss.tar")

    # Make one tensor of predictions and labels list along by concatenating along first dimension
    pred_poss_train = torch.cat(pred_pos_train_list, dim=0)
    label_poss_train = torch.cat(label_pos_train_list, dim=0)
    pred_poss_val = torch.cat(pred_pos_val_list, dim=0)
    label_poss_val = torch.cat(label_pos_val_list, dim=0)

    return history, pred_poss_train.clone().detach(), label_poss_train.clone().detach(), \
           pred_poss_val.clone().detach(), label_poss_val.clone().detach(), model_dir


def main(nr_of_circles, nr_of_sequences_train, nr_of_sequences_val, nr_of_frames_train, nr_of_frames_val, change_shape,
         epochs, batch_size, on_server, lr, lr_scheduling=False):
    save_model = True
    device = get_default_device()

    train_csv_file = ''
    val_csv_file = ''
    if on_server:
        train_csv_file += '/export/home2/NoCsBack/thesisdt/WV/'
        val_csv_file += '/export/home2/NoCsBack/thesisdt/WV/'
    changing = ''
    if change_shape:
        changing = 'changing_'
    train_csv_file += f'datasets/linear_moving_{changing}circles_ff/circles={nr_of_circles}_frames={nr_of_frames_train}' \
                      f'_noise=None/train={nr_of_sequences_train}/train.csv'
    val_csv_file += f'datasets/linear_moving_{changing}circles_ff/circles={nr_of_circles}_frames={nr_of_frames_val}_noise=None/' \
                    f'validation={nr_of_sequences_val}/validation.csv'

    train_ds = CirclesDataset(train_csv_file)
    val_ds = CirclesDataset(val_csv_file)

    train_dl = DataLoader(train_ds, batch_size, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, pin_memory=True)

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    opt_func = torch.optim.Adam

    FFModel = FFNetwork().to(device)

    path_save_model = None
    if save_model:
        path_save_model = ""
        if on_server:
            path_save_model += f"/export/home2/NoCsBack/thesisdt/WV/"
        path_save_model += f"logs/models/pos_ff/{changing}circles={nr_of_circles}/t_{len(train_dl.dl.dataset)}_v_" \
                           f"{len(val_dl.dl.dataset)}_b_{batch_size}"

    print("nr of circles: ", nr_of_circles)
    print("nr of sequences train: ", nr_of_sequences_train)
    print("nr of sequences val: ", nr_of_sequences_val)
    print("changing shape: ", change_shape)
    print("lr: ", lr)
    print("lr sched: ", lr_scheduling)
    print("batch size: ", batch_size)
    print("epochs: ", epochs)

    history, predictions_train, labels_train, predictions_val, labels_val, model_dir = \
        fit(epochs, lr, FFModel, train_dl, val_dl, opt_func, lr_scheduling, path_save_model)
    plot_losses(history)

    train_dir = train_csv_file.replace('train.csv', '')
    val_dir = val_csv_file.replace('validation.csv', '')

    return predictions_train, labels_train, train_dir, predictions_val, labels_val, val_dir, model_dir


if __name__ == "__main__":
    nr_of_sequences_train = 100
    nr_of_sequences_val = 50
    nr_of_frames_train = 100
    nr_of_frames_val = 100
    on_server = True
    R_per_sequence = False

    if len(sys.argv) == 11:
        nr_of_sequences_train, sequence_length_train, nr_of_sequences_val, sequence_length_val, nr_of_circles, \
            change_shape, nr_of_epochs, batch_size, lr, lr_sched = sys.argv[1:]

        predictions_train, positions_train, train_dir, _, _, _, model_dir = \
            main(nr_of_circles=int(nr_of_circles), nr_of_sequences_train=int(nr_of_sequences_train),
                 nr_of_sequences_val=int(nr_of_sequences_val), nr_of_frames_train=int(sequence_length_train),
                 nr_of_frames_val=int(sequence_length_val), change_shape=(change_shape == "change_shape"),
                 epochs=int(nr_of_epochs), batch_size=int(batch_size), on_server=False, lr=float(lr),
                 lr_scheduling=(lr_sched == "lr_sched"))
        Variance.main(predictions_train, positions_train, model_dir, nr_of_sequences_train, sequence_length_train,
                      R_per_sequence)

    else:
        raise Exception("Wrong number of arguments. Following arguments expected: "
                        "<nr_of_sequences_train> <sequence_length_train> <nr_of_sequences_val> "
                        "<sequence_length_val> <nr_of_circles> <change_shape> <nr_of_epochs> <batch_size> <lr> "
                        "<lr_sched>")


