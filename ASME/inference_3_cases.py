import torch
from config import get_config, get_weights_file_path
from train import get_model, get_ds, get_normalize_dataset
from dataset import causal_mask, LSAMDataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def greedy_decode(model, source, source_mask, max_len, device):
    decoder_sos = torch.full((120,), 1.0).to(device)
    decoder_eos = torch.full((120,), 1.0).to(device)

    encoder_output = model.encode(source, source_mask)
    decoder_input = decoder_sos.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 120)
    while True:
        print(decoder_input.size(1))
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        output = model.project(out[:, -1]).unsqueeze(0).to(device)
        decoder_input = torch.cat([decoder_input, output], dim=1)

        if torch.all(output[0] == decoder_eos):
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, max_len, device):
    model.eval()

    # batch size = 1
    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (b, 1, 1, seq_len)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, max_len, device)

            return model_output


def min_max_normalize(two_d_list, min_value, max_value):
    """Helper function"""
    normalized_2d_list = [
        [(item - min_value) / (max_value - min_value) if max_value - min_value else 0 for item in sublist]
        for sublist in two_d_list
    ]
    return normalized_2d_list


def get_planters_dataset(config):
    """ get dataset from csv file and normlize it
     key: source coordinates value: min-max normalized list"""
    path_list, num_layers_list = config["path_list"], config["num_layers_list"]
    max_value = -np.infty
    min_value = np.infty
    dataset = {}

    for layer in range(8, 9):
        value_list = []
        csv_file_path = path_list[2] + f'layer_{layer}.csv'
        df = pd.read_csv(csv_file_path)
        df_max = df.values.max()
        df_min = df.values.min()
        if df_max > max_value:
            max_value = df_max
        if df_min < min_value:
            min_value = df_min

        coordinates = df.columns.tolist()
        key = ' '.join(coordinates)
        for column in df:
            column_list = df[column].tolist()
            value_list.append(column_list)
        dataset[key] = value_list

    normalized_dataset = {key: min_max_normalize(value, min_value, max_value) for key, value in dataset.items()}

    return normalized_dataset


def get_table_dataset(config):
    """ get dataset from csv file and normlize it
     key: source coordinates value: min-max normalized list"""
    path_list, num_layers_list = config["path_list"], config["num_layers_list"]
    max_value = -np.infty
    min_value = np.infty
    dataset = {}

    for layer in range(20, 21):
        value_list = []
        csv_file_path = path_list[1] + f'layer_{layer}.csv'
        df = pd.read_csv(csv_file_path)
        df_max = df.values.max()
        df_min = df.values.min()
        if df_max > max_value:
            max_value = df_max
        if df_min < min_value:
            min_value = df_min

        coordinates = df.columns.tolist()
        key = ' '.join(coordinates)
        for column in df:
            column_list = df[column].tolist()
            value_list.append(column_list)
        dataset[key] = value_list

    normalized_dataset = {key: min_max_normalize(value, min_value, max_value) for key, value in dataset.items()}

    return normalized_dataset


def get_totems_dataset(config):
    """ get dataset from csv file and normlize it
     key: source coordinates value: min-max normalized list"""
    path_list, num_layers_list = config["path_list"], config["num_layers_list"]
    max_value = -np.infty
    min_value = np.infty
    dataset = {}

    for layer in range(50, 51):
        value_list = []
        csv_file_path = path_list[0] + f'layer_{layer}.csv'
        df = pd.read_csv(csv_file_path)
        df_max = df.values.max()
        df_min = df.values.min()
        if df_max > max_value:
            max_value = df_max
        if df_min < min_value:
            min_value = df_min

        coordinates = df.columns.tolist()
        key = ' '.join(coordinates)
        for column in df:
            column_list = df[column].tolist()
            value_list.append(column_list)
        dataset[key] = value_list

    normalized_dataset = {key: min_max_normalize(value, min_value, max_value) for key, value in dataset.items()}

    return normalized_dataset


def get_normalize_dataset(config):
    """ get dataset from csv file and normlize it
     key: source coordinates value: min-max normalized list"""
    path_list, num_layers_list = config["path_list"], config["num_layers_list"]
    max_value = -np.infty
    min_value = np.infty
    dataset = {}

    for path, num_layers in zip(path_list, num_layers_list):
        for layer in range(1, num_layers + 1):
            value_list = []
            csv_file_path = path + f'layer_{layer}.csv'
            df = pd.read_csv(csv_file_path)
            df_max = df.values.max()
            df_min = df.values.min()
            if df_max > max_value:
                max_value = df_max
            if df_min < min_value:
                min_value = df_min

            coordinates = df.columns.tolist()
            key = ' '.join(coordinates)
            for column in df:
                column_list = df[column].tolist()
                value_list.append(column_list)
            dataset[key] = value_list

    normalized_dataset = {key: min_max_normalize(value, min_value, max_value) for key, value in dataset.items()}

    return max_value, min_value


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f'Using device {device}')

    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size()).to(device)
    model_filename = get_weights_file_path(config, f'499')
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    X_max, X_min = get_normalize_dataset(config)


    normal_dataset = get_planters_dataset(config)
    val_ds = LSAMDataset(normal_dataset, tokenizer_src, config['seq_len'])
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    a = run_validation(model, val_dataloader, 744, device)
    a_original = a * (X_max - X_min) + X_min
    a_original = a_original.cpu().numpy()
    np.savetxt('planters_inference.txt', a_original, fmt='%d')

    normal_dataset = get_table_dataset(config)
    val_ds = LSAMDataset(normal_dataset, tokenizer_src, config['seq_len'])
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    a = run_validation(model, val_dataloader, 937, device)
    a_original = a * (X_max - X_min) + X_min
    a_original = a_original.cpu().numpy()
    np.savetxt('table_inference.txt', a_original, fmt='%d')

    normal_dataset = get_totems_dataset(config)
    val_ds = LSAMDataset(normal_dataset, tokenizer_src, config['seq_len'])
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    a = run_validation(model, val_dataloader, 436, device)
    a_original = a * (X_max - X_min) + X_min
    a_original = a_original.cpu().numpy()
    np.savetxt('totems_inference.txt', a_original, fmt='%d')
