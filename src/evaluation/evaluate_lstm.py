# Importing PyTorch for loading and running the trained LSTM model
import torch

# Importing NumPy for storing model probabilities and prediction targets
import numpy as np

# Importing DataLoader for processing validation sequences in batches
from torch.utils.data import DataLoader

# Importing Path for creating and handling project file paths
from pathlib import Path

# Importing evaluation metrics for measuring LSTM classification performance
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# Importing the custom MetroPT sequence dataset
from src.data.sequence_dataset import MetroPTSequenceDataset

# Importing the LSTM failure prediction architecture
from src.models.lstm_model import LSTMFailurePredictor


# Finding the main PredictAgent project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating the path to the scaled validation sensor dataset
VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_scaled.csv"
)


# Creating the path to the validation window information
VALIDATION_WINDOWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_windows.csv"
)


# Creating the path to the best saved LSTM model
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lstm_model_v2.pth"
)


# Selecting Apple Silicon GPU acceleration when available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Displaying the device being used for evaluation
print("Using device:")
print(device)


# Creating the validation sequence dataset
validation_dataset = MetroPTSequenceDataset(
    VALIDATION_DATA_PATH,
    VALIDATION_WINDOWS_PATH
)


# Creating the validation DataLoader
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# Creating the same LSTM architecture that was used during training
model = LSTMFailurePredictor(
    input_size=15,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
).to(device)


# Loading the parameters from the best saved LSTM model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


# Switching the model into evaluation mode
model.eval()


# Creating empty lists for storing validation probabilities and targets
all_probabilities = []
all_targets = []


# Disabling gradient calculations because the model is only being evaluated
with torch.no_grad():

    # Processing validation sequences one batch at a time
    for sequences, targets in validation_loader:

        # Moving the sensor sequences to the selected device
        sequences = sequences.to(device)

        # Passing the sequences through the trained LSTM
        outputs = model(sequences)

        # Converting raw LSTM outputs into probabilities between 0 and 1
        probabilities = torch.sigmoid(outputs)

        # Moving the probabilities back to the CPU and storing them
        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        # Storing the corresponding true prediction targets
        all_targets.extend(
            targets.numpy()
        )


# Converting the collected probabilities into a NumPy array
all_probabilities = np.array(all_probabilities)


# Converting the collected targets into an integer NumPy array
all_targets = np.array(
    all_targets,
    dtype=int
)


# Creating initial binary predictions using the default 0.50 threshold
predictions = (
    all_probabilities >= 0.50
).astype(int)


# Calculating the validation confusion matrix
confusion = confusion_matrix(
    all_targets,
    predictions
)


# Calculating ROC-AUC using the predicted probabilities
roc_auc = roc_auc_score(
    all_targets,
    all_probabilities
)


# Calculating Precision-Recall AUC using the predicted probabilities
pr_auc = average_precision_score(
    all_targets,
    all_probabilities
)


# Displaying the validation confusion matrix
print("\nValidation Confusion Matrix:")
print(confusion)


# Displaying precision, recall, F1-score, and support
print("\nValidation Classification Report:")
print(
    classification_report(
        all_targets,
        predictions,
        digits=4,
        zero_division=0
    )
)


# Displaying the ROC-AUC score
print("Validation ROC-AUC:")
print(roc_auc)


# Displaying the Precision-Recall AUC score
print("\nValidation PR-AUC:")
print(pr_auc)


# Displaying the minimum predicted failure probability
print("\nMinimum predicted probability:")
print(all_probabilities.min())


# Displaying the maximum predicted failure probability
print("\nMaximum predicted probability:")
print(all_probabilities.max())