import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LSAMDataset(Dataset):
    def __init__(self, ds, tokenizer_src, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int32)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int32)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int32)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_text = list(self.ds)[index]   # sequence, string type
        tgt_vector = self.ds[list(self.ds)[index]]  # sequence of vector, [[profile_1], [profile_2], [profile_3]]

        # Transform the coordinate into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # get ids of every coordinates
        # Add sos, eos and padding to each sequence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # Add padding, empty [] to decoder
        dec_num_padding_tokens = self.seq_len - len(tgt_vector) - 1
        dec_padding = torch.zeros(120)  # 120 is profile length

        if enc_num_padding_tokens < 0:
            raise ValueError('Sequence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            (
                torch.full((1, 120), 1.0),
                torch.tensor(tgt_vector, dtype=torch.float32),
                torch.zeros(dec_num_padding_tokens, 120)
            ), dim=0
        )

        label = torch.cat(
            (
                torch.tensor(tgt_vector, dtype=torch.float32),
                torch.full((1, 120), 1.0),
                torch.zeros(dec_num_padding_tokens, 120)
            ), dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": ~ torch.all(decoder_input == dec_padding, dim=1).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "padding": torch.full((self.seq_len, ), len(tgt_vector))
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


