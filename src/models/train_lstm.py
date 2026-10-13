# Importing PyTorch for training and saving the LSTM model
import torch

# Importing DataLoader for loading sensor sequences in batches
from torch.utils.data import DataLoader

# Importing Path for creating and handling project file paths
from pathlib import Path

# Importing the custom MetroPT sequence dataset
from src.data.sequence_dataset import MetroPTSequenceDataset

# Importing the LSTM failure prediction architecture
from src.models.lstm_model import LSTMFailurePredictor


# Finding the main PredictAgent project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating paths to the scaled training and validation datasets
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_scaled.csv"
VALIDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation_scaled.csv"


# Creating paths to the training and validation window metadata
TRAIN_WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "train_windows.csv"
VALIDATION_WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "validation_windows.csv"


# Creating the path where the improved LSTM model will be saved
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model_v2.pth"


# Selecting Apple Silicon GPU acceleration when available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Displaying the device being used for training
print("Using device:")
print(device)


# Creating the training sequence dataset
train_dataset = MetroPTSequenceDataset(
    TRAIN_DATA_PATH,
    TRAIN_WINDOWS_PATH
)


# Creating the validation sequence dataset
validation_dataset = MetroPTSequenceDataset(
    VALIDATION_DATA_PATH,
    VALIDATION_WINDOWS_PATH
)


# Creating the training DataLoader for processing sequences in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)


# Creating the validation DataLoader
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# Creating the LSTM failure prediction model
model = LSTMFailurePredictor(
    input_size=15,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
).to(device)


# Counting the negative training windows
negative_count = (train_dataset.targets == 0).sum()

# Counting the positive training windows
positive_count = (train_dataset.targets == 1).sum()


# Reducing the extreme imbalance weight using the square root of the class ratio
positive_weight = (
    negative_count / positive_count
) ** 0.5


# Converting the positive-class weight into a PyTorch tensor
positive_weight_tensor = torch.tensor(
    [positive_weight],
    dtype=torch.float32,
    device=device
)


# Displaying the calculated positive-class weight
print("\nPositive class weight:")
print(float(positive_weight))


# Creating the weighted binary classification loss function
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=positive_weight_tensor
)


# Creating the Adam optimizer with a smaller learning rate for more stable training
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)


# Setting the maximum number of training epochs
NUM_EPOCHS = 15


# Setting the number of epochs allowed without validation improvement
PATIENCE = 3


# Starting with an infinitely large validation loss
best_validation_loss = float("inf")


# Starting the early-stopping counter
epochs_without_improvement = 0


# Training the LSTM for multiple epochs
for epoch in range(NUM_EPOCHS):

    # Switching the model into training mode
    model.train()

    # Starting the total training loss for the current epoch
    training_loss = 0.0


    # Processing the training sequences one batch at a time
    for sequences, targets in train_loader:

        # Moving the sensor sequences to the selected device
        sequences = sequences.to(device)

        # Moving the prediction targets to the selected device
        targets = targets.to(device)


        # Clearing gradients calculated during the previous batch
        optimizer.zero_grad()


        # Passing the sensor sequences through the LSTM
        outputs = model(sequences)


        # Calculating the weighted binary classification loss
        loss = criterion(
            outputs,
            targets
        )


        # Calculating gradients for all trainable model parameters
        loss.backward()


        # Updating the model parameters using the calculated gradients
        optimizer.step()


        # Adding the current batch loss to the total training loss
        training_loss += loss.item()


    # Calculating the average training loss for the current epoch
    average_training_loss = (
        training_loss / len(train_loader)
    )


    # Switching the model into evaluation mode
    model.eval()

    # Starting the total validation loss for the current epoch
    validation_loss = 0.0


    # Disabling gradient calculations during validation
    with torch.no_grad():

        # Processing the validation sequences one batch at a time
        for sequences, targets in validation_loader:

            # Moving the validation sequences to the selected device
            sequences = sequences.to(device)

            # Moving the validation targets to the selected device
            targets = targets.to(device)


            # Generating validation predictions
            outputs = model(sequences)


            # Calculating validation loss
            loss = criterion(
                outputs,
                targets
            )


            # Adding the current batch loss to the total validation loss
            validation_loss += loss.item()


    # Calculating the average validation loss
    average_validation_loss = (
        validation_loss / len(validation_loader)
    )


    # Displaying the current epoch and its training and validation losses
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"- Training Loss: {average_training_loss:.4f} "
        f"- Validation Loss: {average_validation_loss:.4f}"
    )


    # Checking whether the validation loss improved
    if average_validation_loss < best_validation_loss:

        # Updating the best validation loss
        best_validation_loss = average_validation_loss

        # Resetting the early-stopping counter
        epochs_without_improvement = 0

        # Saving the best-performing model parameters
        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        # Displaying confirmation that the improved model was saved
        print("Best model saved.")


    else:

        # Increasing the counter when validation loss does not improve
        epochs_without_improvement += 1


        # Stopping training when validation loss has not improved for several epochs
        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping activated.")
            break


# Displaying the location of the saved improved LSTM model
print("\nBest LSTM v2 model saved to:")
print(MODEL_PATH)