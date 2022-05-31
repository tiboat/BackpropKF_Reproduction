import csv
import math
import sys
import torch
from torch.utils.data import DataLoader
import Feedforward
import Kalman
from CircleGenerator import CirclesDataset, create_directory, save_tensor
from DeviceDataLoader import DeviceDataLoader, get_default_device
import os
from datetime import datetime
import numpy as np

device = get_default_device()

NOT_SAVING = 0
NONMOVING_SAMPLE = 1
SEQUENCE_SLICING = 2


def get_test_loss(FFModel, KalmanModel, test_loader, loss_function, save_matrices, save_predictions, random_start_pos,
                  random_start_vel, upper_bound_L_hat, simplified_cov_update):
    losss = []
    predictionsDict = dict()
    L_hats = []
    with torch.no_grad():
        FFModel.eval(), KalmanModel.eval()
        total_loss = 0.0
        num_batches = 0
        for i, batch in enumerate(test_loader):
            num_batches += 1
            loss, predictions, L_hat_list = testing_step(FFModel, KalmanModel, batch, loss_function, random_start_pos,
                                                         random_start_vel, upper_bound_L_hat, simplified_cov_update)
            total_loss += loss
            losss.append(loss)

            # Save first amt_frames_to_save tensor frames of sequence of last batch with its label and prediction
            if save_predictions:
                if save_predictions == NONMOVING_SAMPLE:
                    tensors = batch[0][-1][-1]
                    labels = batch[1][-1][-1]
                    predictions = predictions[-1][-1]
                    predictionsDict[i] = (tensors, labels, predictions)

                if save_predictions == SEQUENCE_SLICING:
                    amt_frames_to_save = 50
                    tensors = batch[0][-1][:amt_frames_to_save]
                    labels = batch[1][-1][:amt_frames_to_save]
                    predictions = predictions.permute(1, 0, 2)[-1][:amt_frames_to_save]
                    for j in range(amt_frames_to_save):
                        predictionsDict[i * amt_frames_to_save + j] = (tensors[j], labels[j], predictions[j])
                    save_predictions = 0

            if save_matrices:
                L_hats.append(L_hat_list)

        return total_loss / num_batches, losss, predictionsDict, L_hats


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


def testing_step(FFModel, KalmanModel, batch, loss_function, random_start_pos, random_start_vel, upper_bound_L_hat,
                 simplified_cov_update):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height), labels: (N, T, (x, y, vx, vy) )
    images = batch[0].permute(1, 0, 2, 3, 4)
    positions = batch[1].permute(1, 0, 2)[:, :, :2]
    vels = batch[1].permute(1, 0, 2)[:, :, 2:]

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


def sdv_sqrt_losses(losses):
    # Take square of losses to get root square losses
    sqrt_losses = np.sqrt(losses)
    # Compute variance
    var_sqr_losses = np.var(sqrt_losses.numpy())
    # Return standard deviation
    return np.sqrt(var_sqr_losses)


def test(FFModel, KalmanModel, test_loader, loss_function, on_server, save_matrices, save_predictions,
         random_start_pos, random_start_vel, upper_bound_L_hat, nr_of_circles, simplified_cov_update):
    # Testing phase
    mean_test_loss, losss, predictionsDict, L_hats = \
        get_test_loss(FFModel, KalmanModel, test_loader, loss_function, save_matrices, save_predictions,
                      random_start_pos, random_start_vel, upper_bound_L_hat, simplified_cov_update)
    print("-------------")
    print("Mean test loss")
    print(mean_test_loss.item())
    print("-------------")
    print("Root mean square error (RMSE) ± standard deviation")
    print(f"{np.sqrt(mean_test_loss.item())} ± {sdv_sqrt_losses(torch.tensor(losss, device = 'cpu'))}")

    # Saving Phase
    path = ""
    if on_server:
        path += f"/export/home2/NoCsBack/thesisdt/"
    done = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    path += f"WV/logs/c_{nr_of_circles}_test_{len(test_loader.dl.dataset)}_{done}/"

    if save_predictions or save_matrices:
        create_directory(path)

    if save_matrices:
        save_L_hats_as_csv(L_hats, path)

    if save_predictions:
        save_tensor_label_prediction(predictionsDict, path)


def save_L_hats_as_csv(L_hats, path):
    csv_L_hat = "matrix_L_hat"
    with open(path + csv_L_hat, "w") as csvfile:
        writer = csv.writer(csvfile)
        # Header: epoch | matrix name | gradient
        writer.writerow(["epoch", "L_hat"])
        for L_hat in L_hats:
            writer.writerow(L_hat.cpu().detach().numpy())


def save_tensor_label_prediction(predictionsDict, path):
    csv_name = "tensor_label_pred.csv"
    with open(path + csv_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # Header: label x | label y | prediction x | prediction y
        writer.writerow(["label x", "label y", "prediction x", "prediction y"])
        # Skip the tensor entry
        for i, key in enumerate(predictionsDict):
            save_tensor(predictionsDict[key][0].cpu(), path, f"tensor{i}.pt")
            writer.writerow([predictionsDict[key][1][0].item(), predictionsDict[key][1][1].item(),
                             predictionsDict[key][2][0].item(), predictionsDict[key][2][1].item()])


def start_test(nr_of_circles, nr_of_sequences_test, nr_of_frames_test, batch_size, on_server, save_matrices,
               save_predictions, random_start_pos, random_start_vel, upper_bound_L_hat, simplified_cov_update,
               change_shape, bkf_model_path):
    test_csv_file = ''
    if on_server:
        test_csv_file += '/export/home2/NoCsBack/thesisdt/WV/'
    if save_predictions == NONMOVING_SAMPLE:
        test_csv_file += f'datasets/seq_nonmoving_circles/'
    else:
        changing = ''
        if change_shape:
            changing = 'changing_'
        test_csv_file += f'datasets/linear_moving_{changing}circles_bkf/'
    test_csv_file += f'circles={nr_of_circles}_frames={nr_of_frames_test}_noise=None' \
                     f'/test={nr_of_sequences_test}/test.csv'

    test_ds = CirclesDataset(test_csv_file)
    test_dl = DataLoader(test_ds, batch_size, pin_memory=True)
    test_dl = DeviceDataLoader(test_dl, device)
    loss_func = torch.nn.MSELoss()

    FFModel = Feedforward.FFNetwork().to(device)
    KalmanModel = Kalman.KalmanFilter(device, batch_size).to(device)

    # Load trained BKF model
    checkpoint = torch.load(bkf_model_path, map_location=device)
    FFModel.load_state_dict(checkpoint['ff_state_dict'])
    KalmanModel.load_state_dict(checkpoint['kalman_state_dict'])

    print('bkf model', bkf_model_path)
    print("nr of circles: ", nr_of_circles)
    print("nr of sequences test: ", nr_of_sequences_test)
    print("batch size: ", batch_size)
    print("random_start_pos: ", random_start_pos)
    print("random_start_vel: ", random_start_vel)

    test(FFModel, KalmanModel, test_dl, loss_func, on_server, save_matrices, save_predictions,
         random_start_pos, random_start_vel, upper_bound_L_hat, nr_of_circles, simplified_cov_update)


if __name__ == "__main__":
    save_matrices = False
    save_predictions = NOT_SAVING
    random_start_pos = False
    random_start_vel = True
    upper_bound_L_hat = False
    simplified_cov_update = False

    if len(sys.argv) == 7:
        nr_of_sequences_test, sequence_length_test, nr_of_circles, change_shape, batch_size, bkf_model_path \
            = sys.argv[1:]
        start_test(nr_of_circles=int(nr_of_circles), nr_of_sequences_test=int(nr_of_sequences_test),
                   nr_of_frames_test=int(sequence_length_test), batch_size=int(batch_size), on_server=False,
                   save_matrices=save_matrices, save_predictions=save_predictions, random_start_pos=random_start_pos,
                   random_start_vel=random_start_vel, upper_bound_L_hat=upper_bound_L_hat,
                   simplified_cov_update=simplified_cov_update, change_shape=(change_shape == "change_shape"),
                   bkf_model_path=bkf_model_path)
    else:
        raise Exception("Wrong number of arguments. Following arguments expected: "
                        "<nr_of_sequences_test> <sequence_length_test> <nr_of_circles> <change_shape>  <batch_size>"
                        "<bkf_model_path>")
