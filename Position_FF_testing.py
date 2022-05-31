import sys

import torch
from torch.utils.data import DataLoader
import Feedforward
from BKF_testing import save_tensor_label_prediction
from CircleGenerator import CirclesDataset, create_directory
from DeviceDataLoader import DeviceDataLoader, get_default_device
from datetime import datetime
import numpy as np

device = get_default_device()

NOT_SAVING = 0
NONMOVING_SAMPLE = 1
SEQUENCE_SLICING = 2


def get_test_loss(FFModel, test_loader, loss_function, save_predictions):
    losss = []
    predictionsDict = dict()
    # L_hats = []
    with torch.no_grad():
        FFModel.eval()
        total_loss = 0.0
        num_batches = 0
        for i, batch in enumerate(test_loader):
            num_batches += 1
            loss, predictions = testing_step(FFModel, batch, loss_function)
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
                    tensors = batch[0][0][:amt_frames_to_save]
                    labels = batch[1][0][:amt_frames_to_save]
                    predictions = predictions[:amt_frames_to_save]
                    for j in range(amt_frames_to_save):
                        predictionsDict[i * amt_frames_to_save + j] = (tensors[j], labels[j], predictions[j])
                    save_predictions = 0
        return total_loss / num_batches, losss, predictionsDict


def testing_step(FFModel, batch, loss_function):
    # N is batch size
    # T is amt of frames in sequence
    # Sample dimensions (images, positions), images: (N, T, channels, width, height), labels: (N, T, (x, y, vx, vy))
    images, labels = batch
    N, T, C, W, H = images.shape[0], images.shape[1], images.shape[2], images.shape[3], images.shape[4]
    un_sequenced_batch = images.view(N * T, C, W, H)
    positions = labels[:, :, :2]
    un_sequenced_positions = positions.view(N * T, -1)
    zt, _ = FFModel(un_sequenced_batch)  # Generate predictions
    loss = loss_function(zt, un_sequenced_positions)  # Calculate loss
    return loss, zt


def sdv_sqrt_losses(losses):
    # Take square of losses to get root square losses
    sqrt_losses = np.sqrt(losses)
    # Compute variance
    var_sqr_losses = np.var(sqrt_losses.numpy())
    # Return standard deviation
    return np.sqrt(var_sqr_losses)


def test(FFModel, test_loader, loss_func, on_server, save_predictions, nr_of_circles):
    # Testing phase
    mean_test_loss, losss, predictionsDict = get_test_loss(FFModel, test_loader, loss_func, save_predictions)
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
    path += f"WV/logs/ff_test/c_{nr_of_circles}_test_{len(test_loader.dl.dataset)}_{done}/"

    if save_predictions:
        create_directory(path)
        save_tensor_label_prediction(predictionsDict, path)


def start_test(nr_of_circles, nr_of_sequences_test, nr_of_frames_test, batch_size, on_server, save_predictions,
               ff_model_path, change_shape):
    test_csv_file = ''
    if on_server:
        test_csv_file += '/export/home2/NoCsBack/thesisdt/WV/'
    if save_predictions == NONMOVING_SAMPLE:
        test_csv_file += f'datasets/seq_nonmoving_circles/'
    else:
        changing = ''
        if change_shape:
            changing = 'changing_'
        test_csv_file += f'datasets/linear_moving_{changing}circles_ff/'
    test_csv_file += f'circles={nr_of_circles}_frames={nr_of_frames_test}_noise=None' \
                     f'/test={nr_of_sequences_test}/test.csv'

    test_ds = CirclesDataset(test_csv_file)
    test_dl = DataLoader(test_ds, batch_size, pin_memory=True)
    test_dl = DeviceDataLoader(test_dl, device)
    loss_func = torch.nn.MSELoss()

    FFModel = Feedforward.FFNetwork()

    print("ff_model_path", ff_model_path)
    print("nr_of_sequences_test: ", nr_of_sequences_test)
    print("changing shape: ", change_shape)
    print("nr of circles: ", nr_of_circles)
    print("batch size: ", batch_size)

    # Load trained BKF model
    checkpoint = torch.load(ff_model_path, map_location=device)
    FFModel.load_state_dict(checkpoint['ff_state_dict'])
    FFModel.to(device)

    test(FFModel, test_dl, loss_func, on_server, save_predictions, nr_of_circles)


if __name__ == "__main__":
    if len(sys.argv) == 7:
        nr_of_sequences_test, sequence_length_test, nr_of_circles, change_shape, batch_size, pos_ff_model_path \
            = sys.argv[1:]
        start_test(nr_of_circles=int(nr_of_circles), nr_of_sequences_test=int(nr_of_sequences_test),
                   nr_of_frames_test=int(sequence_length_test), batch_size=int(batch_size), on_server=False,
                   save_predictions=False, ff_model_path=pos_ff_model_path,
                   change_shape=(change_shape == "change_shape"))

    else:
        raise Exception("Wrong number of arguments. Following arguments expected: "
                        "<nr_of_sequences_test> <sequence_length_test> <nr_of_circles> <change_shape> <batch_size> "
                        "<pos_ff_model_path>")
