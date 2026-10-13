# Importing PyTorch for tensor operations
import torch

# Importing neural-network components for building the Transformer model
import torch.nn as nn


# Creating the Transformer model for predicting upcoming compressor failures
class TransformerFailurePredictor(nn.Module):

    def __init__(
        self,
        input_size=15,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):

        # Initializing the parent PyTorch neural-network class
        super().__init__()


        # Converting the 15 sensor values into 64-dimensional embeddings
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )


        # Creating a learnable positional embedding for the 360 time steps
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                360,
                d_model
            )
        )


        # Creating one Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )


        # Stacking multiple Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        # Creating dropout before the final prediction layer
        self.dropout = nn.Dropout(dropout)


        # Creating the final binary failure prediction layer
        self.output_layer = nn.Linear(
            d_model,
            1
        )


    def forward(self, x):

        # Projecting the 15 sensor features into the Transformer embedding space
        x = self.input_projection(x)


        # Adding time-position information to each sensor reading
        x = x + self.position_embedding[:, :x.size(1), :]


        # Passing the complete sensor sequence through the Transformer encoder
        x = self.transformer_encoder(x)


        # Averaging information across all time steps
        x = x.mean(dim=1)


        # Applying dropout before producing the final prediction
        x = self.dropout(x)


        # Generating the raw binary classification score
        prediction = self.output_layer(x)


        # Removing the unnecessary final dimension
        prediction = prediction.squeeze(1)


        # Returning raw logits for BCEWithLogitsLoss
        return prediction