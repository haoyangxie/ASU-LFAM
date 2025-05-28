from pathlib import Path


def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 500,
        "lr": 10**-4,
        "seq_len": 1100,
        "d_model": 256,
        "model_folder": "weights",
        "model_basename": "table_totems_planters_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        "path_list": ['./csv/totems/transformer_input/', './csv/table/transformer_input/', './csv/planters/transformer_input/'],
        "num_layers_list": [89, 46, 13]
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)