import math

from CircleGenerator import load_tensor, plot_frame_and_predictions, plot_frame_and_one_prediction, plot_frame


def plot_frame_and_predictions_locally():
    folder = input("Enter the folder name: ")
    tensor = load_tensor("./Output/" + folder, "tensor.pt")
    plot_frame_and_predictions(tensor, "./Output/" + folder + "/tensor_label_pred.csv")


def plot_multiple_frames_and_predictions_locally():
    folder = input("Enter the folder name: ")
    amount = int(input("Enter number of tensors to plot: "))
    for i in range(amount):
        tensor = load_tensor("./Output/" + folder, f"tensor{i}.pt")
        plot_frame_and_one_prediction(tensor, "./Output/" + folder + "/tensor_label_pred.csv", i)


def plot_sequence_locally(sequenceLength):
    step = sequenceLength // 25
    width = math.floor(math.log(sequenceLength, 10)) + 1
    folder = input("Enter the folder name: ")
    i = 0
    tensor = load_tensor("./Output/" + folder, str(i).rjust(width, '0') + ".pt")
    plot_frame(tensor, str(i))
    i = -1
    while i < sequenceLength - 1:
        i += step
        tensor = load_tensor("./Output/" + folder, str(i).rjust(width, '0') + ".pt")
        plot_frame(tensor, str(i))


if __name__ == "__main__":
    plot_multiple_frames_and_predictions_locally()
