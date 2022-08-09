from torch.nn import Conv1d
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from positional_encoder import PeriodicPositionalEncoding


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int = 478, d_model: int = 512, nhead: int = 2, d_hid: int = 1024, nlayers: int = 4, dropout: float = 0.1, max_len: int = 93):
        super().__init__()
        self.model_type = 'Transformer'
        self.max_len = max_len
        self.ntoken = ntoken
        self.face_encoder = nn.Linear(478 * 3, 40)  # init_face to mel feature as first face
        self.audio_encoder = nn.Linear(40 + 80, d_model) # n_mel = 80
        # self.pos_encoder = PositionalEncoding(d_model, dropout, self.max_len)
        self.pos_encoder = PeriodicPositionalEncoding(d_model, dropout, 30, self.max_len)   # 30 fps
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        self.convertor = torch.nn.Sequential(
            Conv1d(
                in_channels= d_model,
                out_channels= d_model * 4,
                kernel_size= 3,
                padding= (4 - 1) // 2,
                ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            Conv1d(
                in_channels= d_model * 4,
                out_channels= d_model,
                kernel_size= 3,
                padding= (4 - 1) // 2,
                )
        )

        self.decoder = nn.Linear(d_model, ntoken * 3)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, init_face: Tensor, src: Tensor) -> Tensor:
        init_face = init_face.reshape(init_face.shape[0], -1)
        encoded_init_face = self.face_encoder(init_face)
        duplicated_encoded_init_face = encoded_init_face.repeat(self.max_len, 1, 1)
        
        src = torch.cat((src, duplicated_encoded_init_face), 2)
        src = self.audio_encoder(src)   
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src).permute(1, 2, 0)  # output: [92, 2, 512] to [Batch, Dim, Time]
        output = self.convertor(output).permute(2, 0, 1) # [2, 512, 92] to [92, 2, 512]
        decoder_output = self.decoder(output)
        decoder_output = decoder_output.view(decoder_output.size(0), decoder_output.size(1), decoder_output.size(2)//3, 3)

        x_output = decoder_output[:, :, :, 0]
        y_output = decoder_output[:, :, :, 1]
        z_output = decoder_output[:, :, :, 2]
        
        return x_output, y_output, z_output
