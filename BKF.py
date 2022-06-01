import csv
import math
import os
import sys
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import Feedforward
import Kalman
from BKF_testing import start_test
from CircleGenerator import CirclesDataset, create_directory, save_tensor
from DeviceDataLoader import DeviceDataLoader, get_default_device
from datetime import datetime

device = get_default_device()


def get_val_loss(FFModel, KalmanModel, val_loader, loss_function, random_start_pos, random_start_vel, upper_bound_L_hat,
                 simplified_cov_update):
    with torch.no_grad():
        FFModel.eval()
        KalmanModel.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in val_loader:
            num_batches += 1
            loss, _, _ = end_to_end_training_step(FFModel, KalmanModel, batch, loss_function, random_start_pos,
                                                  random_start_vel, upper_bound_L_hat, simplified_cov_update)
            total_loss += loss
        return total_loss / num_batches


def epoch_end(self, epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, last_lr: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss'], result['lr'][-1]))


def epoch_end_no_lr(self, epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss']))


def apply_upper_bound(L_hat_List):
    # L_hat_List tensor of shape (N * T, 3)
    upper_bound = 2.0
    lower_bound = -2.0
    reference_tensor_up = torch.tensor([upper_bound, math.exp(upper_bound), upper_bound]) \
        .repeat(L_hat_List.shape[0], 1).to(device)

    reference_tensor_down = torch.tensor([lower_bound, -math.exp(lower_bound), lower_bound]) \
        .repeat(L_hat_List.shape[0], 1).to(device)

    condition = L_hat_List > reference_tensor_up
    L_hat_List[condition] = upper_bound

    condition = L_hat_List < reference_tensor_down
    L_hat_List[condition] = lower_bound


def end_to_end_training_step(FFModel, KalmanModel, sample_batched, loss_function, random_start_pos, random_start_vel,
                             upper_bound_L_hat, simplified_cov_update):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height), labels: (N, T, (x, y, vx, vy) )
    images = sample_batched[0].permute(1, 0, 2, 3, 4)
    positions = sample_batched[1].permute(1, 0, 2)[:, :, :2]
    vels = sample_batched[1].permute(1, 0, 2)[:, :, 2:]

    # Format data
    # Images in sequence, not including first image (T-1, N, 3, 128, 128)
    images = torch.stack([images[ii] for ii in range(1, len(images))]).float().to(device)

    first_frame_positions = None
    first_frame_vels = None
    if not random_start_pos:
        # First frames of every sequence in batch (N, 2)
        first_frame_positions = positions[0]
    if not random_start_vel:
        # First frames of every sequence in batch (N, 2)
        first_frame_vels = vels[0]

    # (T, N, 2)
    positions = torch.stack([positions[ii] for ii in range(1, len(positions))]).float().to(device)

    # Reshape images so everything can be processed in parallel by utilizing batch size
    T, N, C, W, H = images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]
    seq_images = images.view(T * N, C, W, H)

    # Forward pass
    # output (T * N, dim_output)
    z_list, L_hat_list = FFModel(seq_images)

    if upper_bound_L_hat:
        apply_upper_bound(L_hat_list)

    # Decompress the results into original images format
    z_list = z_list.view(T, N, z_list.shape[1])
    L_hat_list = L_hat_list.view(T, N, L_hat_list.shape[1])

    # Pass through KF
    position_prediction = KalmanModel(z_list, L_hat_list, first_frame_positions, first_frame_vels,
                                      simplified_cov_update)

    loss = loss_function(position_prediction, positions)
    return loss, position_prediction, L_hat_list


def fit(epochs, lr, FFModel, KalmanModel, train_loader, val_loader, opt_func, loss_function, on_server, save_matrices,
        save_predictions, save_lrs, lr_scheduling, random_start_pos, random_start_vel, path_model,
        upper_bound_L_hat, simplified_cov_update):
    history = []
    predictionsDict = dict()
    matricesDict = dict()
    lrsDict = dict()

    optimizer = opt_func(list(FFModel.parameters()) + list(KalmanModel.parameters()), lr)

    sched = None
    lowest_val_loss = None
    loss = None
    model_dict = None

    if lr_scheduling:
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        FFModel.train()
        KalmanModel.train()
        train_losses = []
        lrs = []
        for i, batch in enumerate(train_loader):
            loss, predictions, L_hat_list = end_to_end_training_step(FFModel, KalmanModel, batch, loss_function,
                                                                     random_start_pos,
                                                                     random_start_vel, upper_bound_L_hat,
                                                                     simplified_cov_update)

            train_losses.append(loss)
            loss.backward()
            optimizer.step()

            if lr_scheduling:
                lrs.append(get_lr(optimizer))
                sched.step()

            if i == int(len(train_loader.dl.dataset) / train_loader.dl.batch_size) - 1:
                # Save last tensor of last batch with its label and prediction
                if save_predictions:
                    tensor = batch[0][-1][-1]
                    label = batch[1][-1][-1]
                    prediction = predictions[-1][-1]
                    if "tensor" not in predictionsDict:
                        predictionsDict["tensor"] = tensor
                    predictionsDict[epoch] = (label, prediction)

                if save_matrices:
                    matricesDict[epoch] = ((KalmanModel.diag_Q.clone(),
                                            KalmanModel.diag_Q.grad.clone()), L_hat_list)

                if save_lrs:
                    lrsDict[epoch] = get_lr(optimizer)

                optimizer.zero_grad()

        # Validation phase
        mean_val_loss = get_val_loss(FFModel, KalmanModel, val_loader, loss_function, random_start_pos,
                                     random_start_vel, upper_bound_L_hat, simplified_cov_update)

        if lr_scheduling:
            result = {'train_loss': torch.stack(train_losses).mean().item(), 'val_loss': mean_val_loss, 'lr': lrs}
            epoch_end(epoch, result)
        else:
            result = {'train_loss': torch.stack(train_losses).mean().item(), 'val_loss': mean_val_loss}
            epoch_end_no_lr(epoch, result)

        history.append(result)

        if path_model is not None and epoch > epochs // 2:
            # Save model if lowest val loss
            val_loss = result['val_loss']
            if lowest_val_loss is None or lowest_val_loss > val_loss:
                lowest_val_loss = val_loss
                model_dict = {
                    "epoch": epoch,
                    "ff_state_dict": FFModel.state_dict(),
                    "kalman_state_dict": KalmanModel.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }

    path = ""
    if on_server:
        path += f"/export/home2/NoCsBack/thesisdt/"
    done = datetime.now().strftime("%d-%m-%Y_%H-%M")
    path += f"WV/logs/bkf_logs_paper/t_{len(train_loader.dl.dataset)}_v_{len(val_loader.dl.dataset)}_b_{train_loader.dl.batch_size}_" \
            f"{done}/"

    if save_predictions or save_matrices:
        create_directory(path)

    if save_matrices:
        save_matrices_as_csv(matricesDict, path)

    if save_predictions:
        save_tensor_label_prediction(predictionsDict, path)

    if save_lrs:
        save_lrs_as_csv(lrsDict, path)

    if path_model is not None:
        done = datetime.now().strftime("%d-%m-%Y_%H-%M")
        model_dir = f"{path_model}_loss_{lowest_val_loss}_{done}"
        create_directory(model_dir)

        torch.save(model_dict, f"{model_dir}/lowest_val_loss.tar")

    return history, model_dir + "/lowest_val_loss.tar"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('val_loss').item() for x in history]
    plt.figure()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


def save_matrices_as_csv(matricesDict, path):
    csv_Q = "matrix_Q.csv"
    csv_L_hat = "matrix_L_hat"
    save_matrix_and_gradient_as_csv("Q", path, csv_Q, matricesDict, 0)
    save_matrix_as_csv("L_hat", path, csv_L_hat, matricesDict, 1)


def save_matrix_as_csv(name, path, csv_name, matricesDict, indexDict):
    with open(path + csv_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # Header: epoch | matrix name | gradient
        writer.writerow(["epoch", "matrix " + name])
        for epoch in range(0, len(matricesDict)):
            writer.writerow([epoch, matricesDict[epoch][indexDict].cpu().detach().numpy()])


def save_matrix_and_gradient_as_csv(name, path, csv_name, matricesDict, indexDict):
    with open(path + csv_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # Header: epoch | matrix name | gradient
        writer.writerow(["epoch", "matrix " + name, "gradient"])
        for epoch in range(0, len(matricesDict)):
            writer.writerow([epoch, matricesDict[epoch][indexDict][0].cpu().detach().numpy(),
                             matricesDict[epoch][indexDict][1].cpu().detach().numpy()])


def save_lrs_as_csv(lrsDict, path):
    csv_name = "lrs.csv"
    with open(path + csv_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # Header: epoch | lr
        writer.writerow(["epoch", "lr "])
        for epoch in range(0, len(lrsDict)):
            writer.writerow([epoch, lrsDict[epoch]])


def save_tensor_label_prediction(predictionDict, path):
    save_tensor(predictionDict["tensor"].cpu(), path, "tensor.pt")
    csv_name = "tensor_label_pred.csv"
    with open(path + csv_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # Header: epoch | label x | label y | prediction x | prediction y
        writer.writerow(["epoch", "label x", "label y", "prediction x", "prediction y"])
        # Skip the tensor entry
        for epoch in range(0, len(predictionDict) - 1):
            writer.writerow([epoch, predictionDict[epoch][0][0].item(), predictionDict[epoch][0][1].item(),
                             predictionDict[epoch][1][0].item(), predictionDict[epoch][1][1].item()])


def train_and_validate(nr_of_circles, nr_of_sequences_train, nr_of_sequences_val, nr_of_frames_train, nr_of_frames_val,
                       nr_of_epochs, batch_size, on_server, save_matrices, save_predictions, save_lrs, lr,
                       lr_scheduling,
                       random_start_pos, random_start_vel, static_seq, save_model, upper_bound_L_hat,
                       simplified_cov_update, change_shape, ff_model_path):
    train_csv_file = ''
    val_csv_file = ''
    if on_server:
        train_csv_file += '/export/home2/NoCsBack/thesisdt/WV/'
        val_csv_file += '/export/home2/NoCsBack/thesisdt/WV/'
    if static_seq:
        train_csv_file += f'datasets/seq_nonmoving_circles/'
        val_csv_file += f'datasets/seq_nonmoving_circles/'
    else:
        train_csv_file += f'datasets/linear_moving_circles_bkf/'
        val_csv_file += f'datasets/linear_moving_circles_bkf/'
    train_csv_file += f'circles={nr_of_circles}_frames={nr_of_frames_train}_noise=None' \
                      f'/train={nr_of_sequences_train}/train.csv'
    val_csv_file += f'circles={nr_of_circles}_frames={nr_of_frames_val}_noise=None' \
                    f'/validation={nr_of_sequences_val}/validation.csv'

    print("Training on: ")
    print(train_csv_file)
    print("----------------------------------")

    train_ds = CirclesDataset(train_csv_file)
    val_ds = CirclesDataset(val_csv_file)

    train_dl = DataLoader(train_ds, batch_size, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, pin_memory=True)

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    opt_func = torch.optim.Adam
    loss_func = torch.nn.MSELoss()

    FFModel = Feedforward.FFNetwork().to(device)
    KalmanModel = Kalman.KalmanFilter(device, batch_size).to(device)

    path_save_model = None
    if save_model:
        path_save_model = ""
        if on_server:
            path_save_model += f"/export/home2/NoCsBack/thesisdt/WV/"
        changing = ''
        if change_shape:
            changing = 'changing_'
        path_save_model += f"logs/models/bkf/{changing}circles={nr_of_circles}/t_{len(train_dl.dl.dataset)}_v_" \
                           f"{len(val_dl.dl.dataset)}_b_{batch_size}"

    # Load trained feedforward model
    if ff_model_path is not None:
        checkpoint = torch.load(ff_model_path)
        FFModel.load_state_dict(checkpoint['ff_state_dict'])
        print("Using trained FFModel")
        print(checkpoint['path_training_set'])
        print("----------------------------------")

    print("ff model", ff_model_path)
    print("nr of circles: ", nr_of_circles)
    print("nr of sequences train: ", nr_of_sequences_train)
    print("nr of sequences val: ", nr_of_sequences_val)
    print("lr: ", lr)
    print("lr sched: ", lr_scheduling)
    print("batch size: ", batch_size)
    print("epochs: ", nr_of_epochs)
    print("random_start_pos: ", random_start_pos)
    print("random_start_vel: ", random_start_vel)
    print("change shape: ", change_shape)

    history, model_path = fit(nr_of_epochs, lr, FFModel, KalmanModel, train_dl, val_dl, opt_func, loss_func, on_server,
                              save_matrices, save_predictions, save_lrs, lr_scheduling, random_start_pos,
                              random_start_vel,
                              path_save_model, upper_bound_L_hat, simplified_cov_update)
    plot_losses(history)

    return model_path


if __name__ == "__main__":
    on_server = False
    save_matrices = False
    save_predictions = False
    save_lrs = False
    random_start_pos = False
    random_start_vel = True
    static_seq = False
    save_model = True
    upper_bound_L_hat = False
    simplified_cov_update = False

    if len(sys.argv) == 12:
        nr_of_sequences_train, sequence_length_train, nr_of_sequences_val, sequence_length_val, nr_of_circles, \
        change_shape, nr_of_epochs, batch_size, ff_model_path, lr, lr_sched = sys.argv[1:]
        train_and_validate(nr_of_circles=int(nr_of_circles), nr_of_sequences_train=int(nr_of_sequences_train),
                           nr_of_sequences_val=int(nr_of_sequences_val), nr_of_frames_train=int(sequence_length_train),
                           nr_of_frames_val=int(sequence_length_val), nr_of_epochs=int(nr_of_epochs),
                           batch_size=int(batch_size), on_server=False, save_matrices=save_matrices,
                           save_predictions=save_predictions, save_lrs=save_lrs,
                           change_shape=(change_shape == 'change_shape'), lr=float(lr),
                           lr_scheduling=(lr_sched == 'lr_sched'), random_start_pos=random_start_pos,
                           random_start_vel=random_start_vel, static_seq=static_seq, save_model=True,
                           upper_bound_L_hat=False, simplified_cov_update=False, ff_model_path=ff_model_path)

    else:
        raise Exception("Wrong number of arguments. Following arguments expected: "
                        "<nr_of_sequences_train> <sequence_length_train> <nr_of_sequences_val> "
                        "<sequence_length_val> <nr_of_circles> <change_shape> <nr_of_epochs> <batch_size> "
                        "<ff_model_path> <lr> <lr_sched>")
