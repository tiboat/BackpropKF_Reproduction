import torch

from CircleGenerator import save_tensor


def get_variance_matrix(predictions, positions):
    # predictions: (T*N, 2) or (T, 2)
    # positions: (T*N, 2) or (T, 2)
    errors = predictions.t() - positions.t()
    return torch.cov(errors)


def main(predictions, positions, dir, nr_of_sequences, seq_length, var_per_sequence):
    # predictions: (T*N, 2)
    # positions: (T*N, 2)
    if var_per_sequence:
        R_list = []
        for seq_nr in range(0, nr_of_sequences):
            start_frame_seq = seq_nr * nr_of_sequences
            end_frame_seq = start_frame_seq + seq_length
            seq_predictions = predictions[start_frame_seq:end_frame_seq]
            seq_positions = positions[start_frame_seq:end_frame_seq]
            variance_matrix = get_variance_matrix(seq_predictions, seq_positions)
            R_list.append(variance_matrix)
        R_tensor = torch.stack(R_list)
        save_tensor(R_tensor, dir, "R_seq.pt")
    else:
        variance_matrix = get_variance_matrix(predictions, positions)
        save_tensor(variance_matrix, dir, "R.pt")
