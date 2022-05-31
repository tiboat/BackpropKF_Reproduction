import os
import sys
from datetime import datetime
import torch
from matplotlib import pyplot as plt
from CircleGenerator import CirclesDataset, create_directory
from torch.utils.data import DataLoader
from DeviceDataLoader import DeviceDataLoader, get_default_device
import torch.nn.functional as F
from Feedforward import FFNetwork

device = get_default_device()


def evaluate(model, val_loader, R, R_per_sequence, nr_of_frames_val, batch_size):
    model.eval()
    outputs = [validation_step(model, batch, batch_nr, R, R_per_sequence, nr_of_frames_val, batch_size)
               for batch_nr, batch in enumerate(val_loader)]
    return model.validation_epoch_end(outputs)


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


def process_L_hat_single(L_hat_single):
    # L_hat_single: (3, ) tensor
    # R: (2, 2) tensor
    L = torch.zeros(2, 2)
    L[0, 0] = torch.exp(L_hat_single[0])
    L[1, 0] = L_hat_single[1]
    L[1, 1] = torch.exp(L_hat_single[2])
    R = torch.matmul(L, L.t())
    return R


def process_L_hat_batch(L_hat):
    # L_hat: (N, 3) tensor
    # R: (N, 2, 2) tensor
    N = L_hat.size(0)
    L_hat_tuple = L_hat.unbind(0)
    R_list = [process_L_hat_single(L_hat_single) for L_hat_single in L_hat_tuple]
    R = torch.stack(R_list)
    return R


def get_R_label_flat_scaled(R, R_per_sequence, N, T, batch_nr, frames_per_seq, batch_size):
    R_label_flat = None
    if R_per_sequence:
        start_frame_batch = batch_nr * batch_size
        end_frame_batch = start_frame_batch + batch_size
        R_flat_list = []
        for i in range(start_frame_batch, end_frame_batch):
            R_one_seq = R[i].flatten().repeat(frames_per_seq, 1) * 100
            R_flat_list.append(R_one_seq)
        R_label_flat = torch.cat(R_flat_list, dim=0)
    else:
        R_label_flat = R.flatten().repeat(N * T, 1) * 100
    return R_label_flat


def training_step(model, batch, batch_nr, R, R_per_sequence, frames_per_seq, batch_size):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height),
    # labels: (N, T, (x, y, vx, vy) ) so (N, T, 4)
    images, labels = batch
    N, T, C, W, H = images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]
    un_sequenced_batch = images.view(N*T, C, W, H)
    zt, L_hat = model(un_sequenced_batch)  # Generate predictions

    # Make label with position and variance to compute loss
    positions = labels[:, :, :2]  # (N, T, 2)
    un_sequenced_positions = positions.view(N * T, -1)  # (N*T, 2)
    # R_label_flat = R_label.flatten().repeat(N*T, 1)  # (N*T, 4)
    R_label_flat = get_R_label_flat_scaled(R, R_per_sequence, N, T, batch_nr, frames_per_seq, batch_size)
    label_for_loss = torch.column_stack((un_sequenced_positions, R_label_flat))  # (N*T, 6)

    # Make observation with position and variance to compute loss
    R_pred_flat = process_L_hat_batch(L_hat).flatten(start_dim=1).to(device) * 100  # (N*T, 4)
    obs_for_loss = torch.column_stack((zt, R_pred_flat))  # (N*T, 6)

    loss = F.mse_loss(obs_for_loss, label_for_loss)  # Calculate loss
    return loss, R_pred_flat / 100


def validation_step(model, batch, batch_nr, R, R_per_sequence, frames_per_seq, batch_size):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height),
    # labels: (N, T, (x, y, vx, vy) ) so (N, T, 4)
    images, labels = batch
    N, T, C, W, H = images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]
    un_sequenced_batch = images.view(N * T, C, W, H)
    zt, L_hat = model(un_sequenced_batch)  # Generate predictions

    # Make label with position and variance to compute loss
    positions = labels[:, :, :2]  # (N, T, 2)
    un_sequenced_positions = positions.view(N * T, -1)  # (N*T, 2)
    R_label_flat = get_R_label_flat_scaled(R, R_per_sequence, N, T, batch_nr, frames_per_seq, batch_size)
    label_for_loss = torch.column_stack((un_sequenced_positions, R_label_flat))  # (N*T, 6)

    # Make observation with position and variance to compute loss
    R_pred_flat = process_L_hat_batch(L_hat).flatten(start_dim=1).to(device) * 100  # (N*T, 4)
    obs_for_loss = torch.column_stack((zt, R_pred_flat))  # (N*T, 6)

    loss = F.mse_loss(obs_for_loss, label_for_loss)  # Calculate loss
    return {'val_loss': loss.detach()}


def fit(epochs, lr, model, train_loader, val_loader, opt_func, lr_scheduling, path_model, R,
        R_per_sequence, train_csv_file, nr_of_frames_train, nr_of_frames_val, batch_size):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    sched = None
    loss = None
    model_dict = None
    lowest_val_loss = None
    # Set up one-cycle learning rate scheduler
    if lr_scheduling:
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch_nr, batch in enumerate(train_loader):
            loss, r_pred_flat = training_step(model, batch, batch_nr, R, R_per_sequence, nr_of_frames_train,
                                              batch_size)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduling:
                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()
        # Validation phase
        result = evaluate(model, val_loader, R, R_per_sequence, nr_of_frames_val, batch_size)
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
                    "path_training_set": train_csv_file,
                    "ff_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }

    done = datetime.now().strftime("%d-%m-%Y_%H-%M")
    model_dir = f"{path_model}_loss_{lowest_val_loss}_{done}"
    create_directory(model_dir)

    torch.save(model_dict, f"{model_dir}/lowest_val_loss.tar")

    return history


def main(nr_of_circles, nr_of_sequences_train, nr_of_sequences_val, nr_of_frames_train, nr_of_frames_val, change_shape,
         nr_of_epochs, batch_size, on_server, R_path, lr, lr_scheduling=False):
    R_per_sequence = False
    save_model = True

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

    R = torch.load(R_path, map_location=device)

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
        path_save_model += f"logs/models/ff/{changing}circles={nr_of_circles}/t_{len(train_dl.dl.dataset)}_v_" \
                           f"{len(val_dl.dl.dataset)}_b_{batch_size}"

    print("nr of circles: ", nr_of_circles)
    print("nr of sequences train: ", nr_of_sequences_train)
    print("nr of sequences val: ", nr_of_sequences_val)
    print("changing shape: ", change_shape)
    print("lr: ", lr)
    print("lr sched: ", lr_scheduling)
    print("batch size: ", batch_size)
    print("epochs: ", nr_of_epochs)

    history = fit(nr_of_epochs, lr, FFModel, train_dl, val_dl, opt_func, lr_scheduling, path_save_model,
                  R, R_per_sequence, train_csv_file, nr_of_frames_train, nr_of_frames_val, batch_size)
    plot_losses(history)


if __name__ == "__main__":

    if len(sys.argv) == 12:
        nr_of_sequences_train, sequence_length_train, nr_of_sequences_val, sequence_length_val, nr_of_circles, \
            change_shape, nr_of_epochs, batch_size, R_path, lr, lr_sched = sys.argv[1:]
        main(nr_of_circles=int(nr_of_circles), nr_of_sequences_train=int(nr_of_sequences_train),
             nr_of_sequences_val=int(nr_of_sequences_val), nr_of_frames_train=int(sequence_length_train),
             nr_of_frames_val=int(sequence_length_val), change_shape=(change_shape == "change_shape"),
             nr_of_epochs=int(nr_of_epochs), batch_size=int(batch_size),
             on_server=False, R_path=R_path, lr=float(lr), lr_scheduling=(lr_sched == "lr_sched"))
    else:
        raise Exception("Wrong number of arguments. Following arguments expected: "
                        "<nr_of_sequences_train> <sequence_length_train> <nr_of_sequences_val> "
                        "<sequence_length_val> <nr_of_circles> <change_shape> <nr_of_epochs> <batch_size> <R_path> "
                        "<lr> <lr_sched>")

