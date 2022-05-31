import sys
from random import random
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import numpy as np
import torch
import os
import pathlib
import pandas as pd
from torch.utils.data import Dataset
import csv
from skimage.draw import disk
from skimage.draw import ellipse as draw_ellipse


## CIRCLE CLASSES AND FUNCTIONS


class Circle(object):
    def __init__(self, x, y, r, vx, vy, colour):
        self.x = x
        self.y = y
        self.r = r
        self.vx = vx
        self.vy = vy
        self.colour = colour

    def area(self):
        return np.pi * self.r * self.r

    def mass(self, density):
        return density * 4 * np.pi * (self.r ** 3) / 3


# Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CirclesDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Parent directory for the data.
        """
        self.main_csv = pd.read_csv(csv_file)
        self.transform = transform
        self.data = []

        for i in range(len(self.main_csv)):
            sub_data_dir = self.main_csv.iloc[i, 0]
            sub_csv = pd.read_csv(self.main_csv.iloc[i, 1])
            tensors = []
            labels = []
            for j in range(len(sub_csv)):
                tensor_path = os.path.join(sub_data_dir, sub_csv.iloc[j, 0])
                tensor = torch.load(tensor_path).numpy()
                tensors.append(tensor)
                label = sub_csv.iloc[j, 1:]
                label = np.array(label, dtype=np.float32)
                labels.append(label)
            self.data.append((np.stack(tensors), np.stack(labels)))

    def __len__(self):
        return len(self.data)

    # Gives back one sequence of frames at index idx
    def __getitem__(self, idx):
        return self.data[idx]


# Create a tensor frame with the provided Circle objects
def circles_to_tensor(circles, len_grid=128):
    tensor = torch.zeros([3, len_grid, len_grid])
    for circle in circles:
        (r, g, b) = cls.to_rgb(circle.colour)
        ys, xs = disk((circle.y, circle.x), circle.r, shape=(len_grid, len_grid))
        tensor[0, ys, xs] = r  # red channel
        tensor[1, ys, xs] = g  # green channel
        tensor[2, ys, xs] = b  # blue channel
    return tensor



## ELLIPSE CLASSES AND FUNCTIONS

class Ellipse(object):
    def __init__(self, x, y, rx, ry, vx, vy, vrx, vry, colour):
        """
        :param x: x coordinate of center of ellipse
        :param y: y coordinate of center of ellipse
        :param rx: half of width of ellipse, radius on x-axis
        :param ry: half of height of ellipse, radius on y-axis
        :param vx: velocity of ellipse on x-axis
        :param vy: velocity of ellipse on y-axis
        :param vrx: velocity of rx, change in width of ellipse
        :param vry: velocity of rx, change in height of ellipse
        :param colour: colour of the ellipse
        """
        self.x = x
        self.y = y
        self.rx = rx
        self.ry = ry
        self.vx = vx
        self.vy = vy
        self.vrx = vrx
        self.vry = vry
        self.colour = colour


# Create a tensor frame with the provided Ellipse objects
def ellipses_to_tensor(ellipses, len_grid=128):
    tensor = torch.zeros([3, len_grid, len_grid])
    for ellipse in ellipses:
        (r, g, b) = cls.to_rgb(ellipse.colour)
        ys, xs = draw_ellipse(ellipse.y, ellipse.x, ellipse.ry, ellipse.rx, shape=(len_grid, len_grid))
        tensor[0, ys, xs] = r  # red channel
        tensor[1, ys, xs] = g  # green channel
        tensor[2, ys, xs] = b  # blue channel
    return tensor



# Returns first object in list objects which has same colour as track_colour
def get_track_object(objects, track_colour):
    for obj in objects:
        if cls.same_color(obj.colour, track_colour):
            return obj
    return None



## FILE ORGANIZATION

def save_tensor(tensor, path, filename):
    torch.save(tensor, f"{path}/{filename}")


def load_tensor(path, filename):
    return torch.load(f"{path}/{filename}")


def create_directory(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def save_csv_for_tensors(csv_path_and_name, tensors_filenames, labels):
    with open(csv_path_and_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header: tensor_file | x | y | vx | vy
        writer.writerow(["tensor_file", "x", "y", "vx", "vy"])
        for i in range(len(tensors_filenames)):
            writer.writerow([tensors_filenames[i]] + labels[i])


def save_csv_for_dirs(csv_path_and_name, dirs, sub_csvs):
    with open(csv_path_and_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header: dirs | sub_csvs
        writer.writerow(["dirs", "sub_csvs"])
        for i in range(len(dirs)):
            writer.writerow([dirs[i], sub_csvs[i]])


## PLOTTING

# Plot a list of Circle objects (Deprecated)
def plot_circles(circles, len_grid=128, indicate_track=True, track_colour='r'):
    figure, axes = plt.subplots()
    for circle in circles:
        if cls.same_color(circle.colour, track_colour) and indicate_track:
            # Indicate circle to track
            axes.add_patch(
                plt.Rectangle(
                    (circle.x - 2 * circle.r, circle.y - 2 * circle.r),
                    4 * circle.r, 4 * circle.r, edgecolor='k', fill=False
                ))
        # Draw circle
        axes.add_patch(plt.Circle((circle.x, circle.y), circle.r, color=circle.colour))

    plt.ylim([0, len_grid])
    plt.xlim([0, len_grid])
    plt.show()


# Visualise a tensor
def plot_frame(tensor, title=""):
    plt.imshow(tensor.permute(1, 2, 0))
    plt.title(title)
    plt.show()


# Visualise a tensor and add prediction markings to it
def plot_frame_and_predictions(tensor, predictionsPath):
    plt.imshow(tensor.permute(1, 2, 0))

    xyCsv = pd.read_csv(predictionsPath)
    for i in range(len(xyCsv)):
        if not i % 20:
            xy = (xyCsv.iloc[i, 3], xyCsv.iloc[i, 4])
            # xytext plaatst tekst op de pixel
            plt.annotate(str(i), xy, xytext=(xy[0], xy[1]), color="white")
    plt.show()


def plot_frame_and_one_prediction(tensor, predictionsPath, rowNumber):
    plt.imshow(tensor.permute(1, 2, 0))

    xyCsv = pd.read_csv(predictionsPath)
    xy = (xyCsv.iloc[rowNumber, 2], xyCsv.iloc[rowNumber, 3])
    # xytext plaatst x op de pixel
    plt.annotate("X", xy, xytext=(xy[0], xy[1]), color="white")
    predictionsPath = os.path.dirname(predictionsPath)
    plt.savefig(f"{predictionsPath}/plot{rowNumber}.png")
    plt.show()


## GENERATING DATASETS

#           Grid
#   len  ^
#        |
#        |
#        |
#        |___________>
#                  len

def generate_circles(nr_of_circles, nr_of_frames=1, len_grid=128, min_r=3, max_r=6, track_colour='r'):
    circles = []
    for i in range(nr_of_circles):
        x = np.random.uniform(low=max_r, high=len_grid - max_r)
        y = np.random.uniform(low=max_r, high=len_grid - max_r)
        r = np.random.uniform(low=min_r, high=max_r)
        vx = np.random.uniform(low=-1, high=1)
        vy = np.random.uniform(low=-1, high=1)

        if i == 0:
            # First (= furthest) circle is track circle
            colour = track_colour

            if nr_of_frames > 1:
                # Make sure the track circle doesn't move out of frame
                end_pos_x = x + vx * nr_of_frames
                end_pos_y = y + vy * nr_of_frames
                while end_pos_x < r or end_pos_y < r or end_pos_x > len_grid - r or end_pos_y > len_grid - r:
                    x = np.random.uniform(low=max_r, high=len_grid - max_r)
                    y = np.random.uniform(low=max_r, high=len_grid - max_r)
                    end_pos_x = x + vx * nr_of_frames
                    end_pos_y = y + vy * nr_of_frames
        else:
            colour = (np.random.uniform(), np.random.uniform(), np.random.uniform())
            while colour == track_colour:
                colour = (np.random.uniform(), np.random.uniform(), np.random.uniform())

        circles.append(Circle(x, y, r, vx, vy, colour))

    return circles


def move_circles(circles, noise_pos=None, dt=1):
    for circle in circles:
        old_pos = np.array([circle.x, circle.y])
        v = np.array([circle.vx, circle.vy])

        # linear movement: new_pos = old_pos + velocity * time
        new_pos = old_pos + v * dt

        # add Gaussian noise
        if noise_pos:
            noise = np.random.normal(0, noise_pos, 2)
            new_pos += noise

        # update position
        circle.x, circle.y = new_pos[0], new_pos[1]


def generate_datasets_nonmoving_circles(nr_of_datasets, nr_of_circles, type, on_server, len_grid=128, min_r=3, max_r=6,
                                        track_colour='r'):
    path = ""
    if on_server:
        path += "/export/home2/NoCsBack/thesisdt/WV/"
    path += f"datasets/nonmoving_circles/circles={nr_of_circles}/{type}={nr_of_datasets}/"
    create_directory(path)
    file_names = []
    labels = []
    digits_datasets = len(str(nr_of_datasets))
    for i in range(nr_of_datasets):
        # Generate tensor
        circles = generate_circles(nr_of_circles, len_grid=len_grid, min_r=min_r, max_r=max_r,
                                   track_colour=track_colour)
        tensor = circles_to_tensor(circles, len_grid=len_grid)

        # Determine label
        track_circle = get_track_object(circles, track_colour)
        labels.append([track_circle.x, track_circle.y, track_circle.vx, track_circle.vy])

        file_names.append(str(i).zfill(digits_datasets) + ".pt")

        # Save frame
        save_tensor(tensor, path, file_names[i])

    # Save in .csv
    save_csv_for_tensors(path + f"circles={nr_of_circles}.csv", file_names, labels)


def generate_datasets_sequences_nonmoving_circles(nr_of_datasets, nr_of_circles, type, nr_of_frames, on_server,  model_type,
                                                  noise_pos=None, len_grid=128, min_r=3, max_r=6, track_colour='r'):
    parent_dir = ""
    if on_server:
        parent_dir += "/export/home2/NoCsBack/thesisdt/WV/"
    parent_dir += f"datasets/seq_nonmoving_circles/circles={nr_of_circles}_frames={nr_of_frames}_noise={noise_pos}/" \
                  f"{type}={nr_of_datasets}/"
    create_directory(parent_dir)
    digits_datasets = len(str(nr_of_datasets))
    dirs = []
    sub_csvs = []
    for i in range(nr_of_datasets):  # datasets
        path = parent_dir + str(i).zfill(digits_datasets) + "/"
        create_directory(path)
        dirs.append(path)
        circles = generate_circles(nr_of_circles, nr_of_frames=nr_of_frames, len_grid=len_grid, min_r=min_r,
                                   max_r=max_r, track_colour=track_colour)
        tensor = circles_to_tensor(circles, len_grid=len_grid)

        # Save frame
        file_name = "0.pt"
        save_tensor(tensor, path, file_name)

        # Determine label
        track_circle = get_track_object(circles, track_colour)
        label = [track_circle.x, track_circle.y, 0, 0]

        # Save one sequence in .csv
        csv_name = path + str(i).zfill(digits_datasets) + ".csv"
        sub_csvs.append(csv_name)
        save_csv_for_tensors(csv_name, [file_name] * nr_of_frames, [label] * nr_of_frames)

    # Save references to all sequences in .csv
    save_csv_for_dirs(parent_dir + f"{type}.csv", dirs, sub_csvs)


def generate_datasets_linear_moving_circles(nr_of_datasets, nr_of_circles, type, nr_of_frames, on_server,  model_type, noise_pos=None,
                                            len_grid=128, min_r=3, max_r=6, track_colour='r'):
    parent_dir = ""
    if on_server:
        parent_dir += "/export/home2/NoCsBack/thesisdt/WV/"
    parent_dir += f"datasets/linear_moving_circles_{model_type}/circles={nr_of_circles}_frames={nr_of_frames}_noise={noise_pos}/" \
                  f"{type}={nr_of_datasets}/"
    create_directory(parent_dir)
    digits_datasets = len(str(nr_of_datasets))
    digits_frames = len(str(nr_of_frames))
    dirs = []
    sub_csvs = []
    for i in range(nr_of_datasets):  # datasets
        path = parent_dir + str(i).zfill(digits_datasets) + "/"
        create_directory(path)
        dirs.append(path)
        circles = generate_circles(nr_of_circles, nr_of_frames=nr_of_frames, len_grid=len_grid, min_r=min_r,
                                   max_r=max_r, track_colour=track_colour)
        file_names = []
        labels = []
        for j in range(nr_of_frames):  # frames in datasets
            if j != 0:
                move_circles(circles, noise_pos=noise_pos)
            tensor = circles_to_tensor(circles, len_grid=len_grid)

            # Determine label
            track_circle = get_track_object(circles, track_colour)
            labels.append([track_circle.x, track_circle.y, track_circle.vx, track_circle.vy])

            # Save frame
            file_names.append(str(j).zfill(digits_frames) + ".pt")
            save_tensor(tensor, path, file_names[j])

        # Save one sequence in .csv
        csv_name = path + str(i).zfill(digits_datasets) + ".csv"
        sub_csvs.append(csv_name)
        save_csv_for_tensors(csv_name, file_names, labels)

    # Save references to all sequences in .csv
    save_csv_for_dirs(parent_dir + f"{type}.csv", dirs, sub_csvs)


def generate_datasets_varying_amt_linear_moving_circles(nr_of_datasets, nr_of_circles_lower_bound,
                                                        nr_of_circles_upper_bound, type, nr_of_frames, on_server,  model_type,
                                                        noise_pos=None, len_grid=128, min_r=3, max_r=6,
                                                        track_colour='r'):
    parent_dir = ""
    if on_server:
        parent_dir += "/export/home2/NoCsBack/thesisdt/WV/"
    parent_dir += f"datasets/linear_moving_circles_{model_type}/circles={nr_of_circles_lower_bound}-{nr_of_circles_upper_bound}_" \
                  f"frames={nr_of_frames}_noise={noise_pos}/{type}={nr_of_datasets}/"
    create_directory(parent_dir)
    digits_datasets = len(str(nr_of_datasets))
    digits_frames = len(str(nr_of_frames))
    dirs = []
    sub_csvs = []
    for i in range(nr_of_datasets):  # datasets
        path = parent_dir + str(i).zfill(digits_datasets) + "/"
        create_directory(path)
        dirs.append(path)
        nr_of_circles = np.random.randint(low=nr_of_circles_lower_bound, high=nr_of_circles_upper_bound + 1)
        circles = generate_circles(nr_of_circles, nr_of_frames=nr_of_frames, len_grid=len_grid, min_r=min_r,
                                   max_r=max_r, track_colour=track_colour)
        file_names = []
        labels = []
        for j in range(nr_of_frames):  # frames in datasets
            if j != 0:
                move_circles(circles, noise_pos=noise_pos)
            tensor = circles_to_tensor(circles, len_grid=len_grid)

            # Determine label
            track_circle = get_track_object(circles, track_colour)
            labels.append([track_circle.x, track_circle.y, track_circle.vx, track_circle.vy])

            # Save frame
            file_names.append(str(j).zfill(digits_frames) + ".pt")
            save_tensor(tensor, path, file_names[j])

        # Save one sequence in .csv
        csv_name = path + str(i).zfill(digits_datasets) + ".csv"
        sub_csvs.append(csv_name)
        save_csv_for_tensors(csv_name, file_names, labels)

    # Save references to all sequences in .csv
    save_csv_for_dirs(parent_dir + f"{type}.csv", dirs, sub_csvs)


def generate_ellipses(nr_of_ellipses, nr_of_frames=1, len_grid=128, min_r=2, max_r=10, track_colour='r'):
    ellipses = []
    extra_range = 4
    for i in range(nr_of_ellipses):
        x = np.random.uniform(low=max_r, high=len_grid - max_r)
        y = np.random.uniform(low=max_r, high=len_grid - max_r)
        rx = np.random.uniform(low=min_r+extra_range, high=max_r-extra_range)
        ry = np.random.uniform(low=min_r+extra_range, high=max_r-extra_range)
        vx = np.random.uniform(low=-1, high=1)
        vy = np.random.uniform(low=-1, high=1)
        if random() < 0.5:
            low = (min_r - rx) / nr_of_frames
            vrx = np.random.uniform(low=low, high=low / 2)
        else:
            high = (max_r - rx) / nr_of_frames
            vrx = np.random.uniform(low=high / 2, high=high)
        if random() < 0.5:
            low = (min_r - ry) / nr_of_frames
            vry = np.random.uniform(low=low, high=low / 2)
        else:
            high = (max_r - ry) / nr_of_frames
            vry = np.random.uniform(low=high / 2, high=high)

        if i == 0:
            # First (= furthest) ellipse is track ellipse
            colour = track_colour

            if nr_of_frames > 1:
                # Make sure the track ellipse doesn't move out of frame
                end_pos_x = x + vx * nr_of_frames
                end_pos_y = y + vy * nr_of_frames
                while end_pos_x < rx or end_pos_y < ry or end_pos_x > len_grid - rx or end_pos_y > len_grid - ry:
                    x = np.random.uniform(low=max_r, high=len_grid - max_r)
                    y = np.random.uniform(low=max_r, high=len_grid - max_r)
                    end_pos_x = x + vx * nr_of_frames
                    end_pos_y = y + vy * nr_of_frames
        else:
            colour = (np.random.uniform(), np.random.uniform(), np.random.uniform())
            while colour == track_colour:
                colour = (np.random.uniform(), np.random.uniform(), np.random.uniform())

        ellipses.append(Ellipse(x, y, rx, ry, vx, vy, vrx, vry, colour))

    return ellipses


def move_ellipses(ellipses, noise_pos=None, dt=1):
    for ellipse in ellipses:
        # linear movement: new_pos = old_pos + velocity * time
        old_pos = np.array([ellipse.x, ellipse.y])
        v = np.array([ellipse.vx, ellipse.vy])
        new_pos = old_pos + v * dt

        # change radius x and y
        old_radius = np.array([ellipse.rx, ellipse.ry])
        vr = np.array([ellipse.vrx, ellipse.vry])
        new_radius = old_radius + vr * dt

        # add Gaussian noise
        if noise_pos:
            noise = np.random.normal(0, noise_pos, 2)
            new_pos += noise

        # update position and radius
        ellipse.x, ellipse.y = new_pos[0], new_pos[1]
        ellipse.rx, ellipse.ry = new_radius[0], new_radius[1]


def generate_datasets_linear_moving_ellipses(nr_of_datasets, nr_of_ellipses, type, nr_of_frames, on_server,  model_type,
                                             noise_pos=None, len_grid=128, min_r=1, max_r=10, track_colour='r'):
    parent_dir = ""
    if on_server:
        parent_dir += "/export/home2/NoCsBack/thesisdt/WV/"
    parent_dir += f"datasets/linear_moving_changing_circles_{model_type}/circles={nr_of_ellipses}_frames={nr_of_frames}" \
                  f"_noise={noise_pos}/{type}={nr_of_datasets}/"
    create_directory(parent_dir)
    digits_datasets = len(str(nr_of_datasets))
    digits_frames = len(str(nr_of_frames))
    dirs = []
    sub_csvs = []
    for i in range(nr_of_datasets):  # datasets
        path = parent_dir + str(i).zfill(digits_datasets) + "/"
        create_directory(path)
        dirs.append(path)
        ellipses = generate_ellipses(nr_of_ellipses, nr_of_frames=nr_of_frames, len_grid=len_grid, min_r=min_r,
                                     max_r=max_r, track_colour=track_colour)
        file_names = []
        labels = []
        for j in range(nr_of_frames):  # frames in datasets
            if j != 0:
                move_ellipses(ellipses, noise_pos=noise_pos)
            tensor = ellipses_to_tensor(ellipses, len_grid=len_grid)

            # Determine label
            track_ellipse = get_track_object(ellipses, track_colour)
            labels.append([track_ellipse.x, track_ellipse.y, track_ellipse.vx, track_ellipse.vy])

            # Save frame
            file_names.append(str(j).zfill(digits_frames) + ".pt")
            save_tensor(tensor, path, file_names[j])

        # Save one sequence in .csv
        csv_name = path + str(i).zfill(digits_datasets) + ".csv"
        sub_csvs.append(csv_name)
        save_csv_for_tensors(csv_name, file_names, labels)

    # Save references to all sequences in .csv
    save_csv_for_dirs(parent_dir + f"{type}.csv", dirs, sub_csvs)


if __name__ == "__main__":
    if len(sys.argv) == 7:
        nr_of_datasets, seq_length, nr_of_circles, type, model, change_shape = sys.argv[1:]
        if change_shape != 'change_shape':
            generate_datasets_linear_moving_circles(nr_of_datasets=int(nr_of_datasets),
                                                    nr_of_circles=int(nr_of_circles), type=type,
                                                    nr_of_frames=int(seq_length), on_server=False, model_type=model)
        else:
            generate_datasets_linear_moving_ellipses(nr_of_datasets=int(nr_of_datasets),
                                                     nr_of_ellipses=int(nr_of_circles), type=type,
                                                     nr_of_frames=int(seq_length), on_server=False, model_type=model)
    else:
        raise Exception("Wrong number of arguments. Following arguments expected: "
                        "<nr_of_sequences> <sequence_length> <nr_of_circles> <type> <model> <change_shape>")
