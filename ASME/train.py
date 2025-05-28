import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
from config import get_config
from dataset import LSAMDataset
from model import build_transformer
from config import get_weights_file_path
import warnings


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    return 0


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()


def get_all_sentence(config):
    """Get all coordinates sequence"""
    path_list, num_layers_list = config["path_list"], config["num_layers_list"]
    res = []
    for path, num_layers in zip(path_list, num_layers_list):
        for layer in range(1, num_layers + 1):
            csv_file_path = path + f'layer_{layer}.csv'
            df = pd.read_csv(csv_file_path)
            coordinates = df.columns.tolist()
            result_string = ' '.join(coordinates)
            res.append(result_string)
    return res


def get_or_build_coordinates_tokenizer(config):
    """Generate or get tokenizer json file of all coordinates"""
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
        tokenizer.train_from_iterator(get_all_sentence(config), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def min_max_normalize(two_d_list, min_value, max_value):
    """Helper function"""
    normalized_2d_list = [
        [(item - min_value) / (max_value - min_value) if max_value - min_value else 0 for item in sublist]
        for sublist in two_d_list
    ]
    return normalized_2d_list


def get_normalize_dataset(config):
    """ get dataset from csv file and normlize it
     key: source coordinates value: min-max normalized list"""
    path_list, num_layers_list = config["path_list"], config["num_layers_list"]
    max_value = -np.infty
    min_value = np.infty
    dataset = {}

    for path, num_layers in zip(path_list, num_layers_list):
        # use planter layer 8, table layer 20, totem layer 40 as inference
        for layer in range(1, num_layers + 1):
            if path == './csv/planters/transformer_input/' and layer == 8:
                print('skip planters layer 8')
                continue
            if path == './csv/table/transformer_input/' and layer == 20:
                print('skip table layer 20')
                continue
            if path == './csv/totems/transformer_input/' and layer == 50:
                print('skip totems layer 50')
                continue
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

    return normalized_dataset


def get_ds(config):
    """Get train_dataloader, val_dataloader and coordinates tokenizer_src"""
    ds_raw = get_normalize_dataset(config)

    tokenizer_src = get_or_build_coordinates_tokenizer(config)

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = LSAMDataset(train_ds_raw.dataset, tokenizer_src, config['seq_len'])
    val_ds = LSAMDataset(val_ds_raw.dataset, tokenizer_src, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src


def get_model(config, vocab_src_len):
    """Get transformer model"""
    model = build_transformer(vocab_src_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def my_mse_loss(source, target, seq_len_list):
    batch_size, _, _ = source.shape
    total_loss = 0.0

    for i in range(batch_size):
        valid_length = seq_len_list[i]
        valid_source = source[i, :valid_length]
        valid_target = target[i, :valid_length]
        mse_loss = (valid_target - valid_source).pow(2).mean()
        total_loss += mse_loss

    average_loss = total_loss / batch_size

    return average_loss



def train_model(config):
    # Define the device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    loss_fn = nn.MSELoss(reduction='mean')

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len, 120)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
            trimmed_index = batch['padding'].to(device) # (B, seq_len)
            seq_len_list = trimmed_index[:, 0]

            # run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, 120)
            label = batch['label'].to(device)  # (B, seq_len, 120)
            # proj_output_trimmed = proj_output[:, :label.size(1), :]

            # loss = loss_fn(proj_output, label)
            loss = my_mse_loss(proj_output, label, seq_len_list)
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1

        if global_step % 50 == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
