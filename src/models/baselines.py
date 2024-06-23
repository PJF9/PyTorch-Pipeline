import torch
import torch.nn as nn


class TransformerEncoderAll(nn.Module):
    '''
    A Transformer Encoder model for sequential data processing and classification.
    '''

    def __init__(self,
            window: int,
            input_size: int=5,
            d_model: int=64,
            nhead: int=8,
            num_layers: int=3,
            dim_feedforward: int=128,
            out_features: int=64,
            activation: str='gelu',
            dropout: int=0.2
        ) -> None:
        '''
        Initializes the TransformerEncoderAll model.

        Args:
            window (int): The size of the historical window.
            input_size (int, optional): The size of the input features. Default is 5.
            d_model (int, optional): The number of expected features in the encoder inputs. Default is 64.
            nhead (int, optional): The number of heads in the multihead attention models. Default is 8.
            num_layers (int, optional): The number of sub-encoder-layers in the encoder. Default is 3.
            dim_feedforward (int, optional): The dimension of the feedforward network model. Default is 128.
            out_features (int, optional): The number of output features. Default is 64.
            activation (str, optional): The activation function of the intermediate layer. Default is 'gelu'.
            dropout (int, optional): The dropout value. Default is 0.2.
        '''
        super().__init__()

        self.encode_inputs = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=d_model),
            nn.ReLU()
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=num_layers
        )

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=d_model*window, out_features=out_features),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=out_features, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, window, input_size).

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        '''
        # Encoding input to d_model dimensions
        x = self.encode_inputs(x)

        # Generating logits
        x = self.transformer(x)

        return self.classification_head(x)
