#!/usr/bin/env python
# coding: utf-8

#%% 
# 1
import numpy as np
import matplotlib.pyplot as plt
import librosa 
import pandas as pd
import glob
import os
import fnmatch
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


#%% 
# 2

#make new folders for each classification
new_path = '../input/Heartbeat_Sounds/'
os.makedirs(new_path, exist_ok=True)
unlabel_path = new_path + 'unlabel/'
os.makedirs(unlabel_path, exist_ok=True)
normal_path = new_path + 'normal/'
os.makedirs(normal_path, exist_ok=True)
murmur_path = new_path + 'murmur/'
os.makedirs(murmur_path, exist_ok=True)
extrastole_path = new_path + 'extrastole/'
os.makedirs(extrastole_path, exist_ok=True)
artifact_path = new_path + 'artifact/'
os.makedirs(artifact_path, exist_ok=True)
extrahls_path = new_path + 'extrahls/'
os.makedirs(extrahls_path, exist_ok=True)


#%% 
# 3

#move all files into one folder
set_a_path = '../input/set_a/'
set_a_files = glob.glob(set_a_path + '*.wav')
set_b_path = '../input/set_b/'
set_b_files = glob.glob(set_b_path + '*.wav')
all_files = set_a_files + set_b_files
all_files

for i in all_files:
    filename = i.split('\\')[1]
    os.rename(i,new_path+filename)


#%% 
# 4

#sort files into their respective classifications
unlabel_files = glob.glob(new_path + '*unlabel*.wav')
for i in unlabel_files:
    filename = i.split('\\')[1]
    os.rename(i,unlabel_path+filename)

normal_files = glob.glob(new_path + '*normal*.wav')
for i in normal_files:
    filename = i.split('\\')[1]
    os.rename(i,normal_path+filename)
    
murmur_files = glob.glob(new_path + '*murmur*.wav')
for i in murmur_files:
    filename = i.split('\\')[1]
    os.rename(i,murmur_path+filename)
    
extrastole_files = glob.glob(new_path + '*extrastole*.wav')
for i in extrastole_files:
    filename = i.split('\\')[1]
    os.rename(i,extrastole_path+filename)
    
artifact_files = glob.glob(new_path + '*artifact*.wav')
for i in artifact_files:
    filename = i.split('\\')[1]
    os.rename(i,artifact_path+filename)
    
extrahls_files = glob.glob(new_path + '*extrahls*.wav')
for i in extrahls_files:
    filename = i.split('\\')[1]
    os.rename(i,extrahls_path+filename)


#%% 
# 5

CLASSES = ['normal', 'extrastole', 'extrahls','murmur','artifact']
NB_CLASSES=len(CLASSES)

# Map integer value to text labels
label_to_int = {k:v for v,k in enumerate(CLASSES)}
print (label_to_int)
print (" ")
int_to_label = {v:k for k,v in label_to_int.items()}
print(int_to_label)


#%% 
# 6

def compute_mfcc(y, sr, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Compute MFCCs and return an array of shape (frames, n_mfcc).
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    return mfcc.T.astype(np.float32)

def augment_noise(y, noise_factor=0.005):
    """Add random Gaussian noise."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def augment_stretch(y, rate=0.9):
    """Time‑stretch the signal by the given rate (e.g. 0.9 slows by 10%)."""
    return librosa.effects.time_stretch(y, rate=rate)

def augment_shift(y, shift_max=0.2):
    """Circularly shift (roll) the signal by up to ±shift_max * len(y) samples."""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def pad_or_truncate_audio(y, target_len):
    """
    If y is longer than target_len, truncate; if shorter, pad with zeros.
    """
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), 'constant')
    return y

def extract_and_augment_mfcc_array(
    wav_path,
    sr=22050,
    duration=10.0,
    n_mfcc=13,
    hop_length=512,
    n_fft=2048,
    noise_factor=0.005,
    stretch_rate=0.9,
    shift_max=0.2
):
    """
    Load a WAV, resample to `sr`, pad/truncate to `duration` seconds,
    then compute MFCCs for:
      0: original
      1: + noise
      2: + time‑stretch
      3: + time‑shift

    Returns:
        numpy array of shape (4, frames, n_mfcc)
    """
    # 1) load & resample
    y, orig_sr = librosa.load(wav_path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    # 2) enforce fixed length in samples
    target_len = int(sr * duration)
    y = pad_or_truncate_audio(y, target_len)

    # 3) compute MFCCs
    mfcc_orig    = compute_mfcc(y, sr, n_mfcc, hop_length, n_fft)
    ###mfcc_orig    = mfcc_orig.reshape([-1,1])
    mfcc_noise   = compute_mfcc(pad_or_truncate_audio(augment_noise(y, noise_factor),   target_len), sr, n_mfcc, hop_length, n_fft)
    ###mfcc_noise   = mfcc_noise.reshape([-1,1])
    mfcc_stretch = compute_mfcc(pad_or_truncate_audio(augment_stretch(y, stretch_rate), target_len), sr, n_mfcc, hop_length, n_fft)
    ###mfcc_stretch = mfcc_stretch.reshape([-1,1])
    mfcc_shift   = compute_mfcc(pad_or_truncate_audio(augment_shift(y, shift_max),      target_len), sr, n_mfcc, hop_length, n_fft)
    ###mfcc_shift   = mfcc_shift.reshape([-1,1])

    # 4) stack into single array (4 × frames × n_mfcc)
    #return np.stack([mfcc_orig, mfcc_noise, mfcc_stretch, mfcc_shift], axis=0)
    return [mfcc_orig, mfcc_noise, mfcc_stretch, mfcc_shift]



#%% 
# 7

#Extracting MFCCs from all of the LABELED cases
normal_files=glob.glob(normal_path+'*.wav')
normal_MFCC = []
for i in range(len(normal_files)):
    X = extract_and_augment_mfcc_array(
            normal_files[i],
            sr=22050,
            duration=10.0,
            n_mfcc=50,
            noise_factor=0.01,
            stretch_rate=1.2,
            shift_max=0.2)
    for j in X:
        normal_MFCC.append(j)
normal_labels = [0 for items in normal_MFCC]       

extrastole_files=glob.glob(extrastole_path+'*.wav')
extrastole_MFCC = []
for i in range(len(extrastole_files)):
    X = extract_and_augment_mfcc_array(
            extrastole_files[i],
            sr=22050,
            duration=10.0,
            n_mfcc=50,
            noise_factor=0.01,
            stretch_rate=1.2,
            shift_max=0.2)
    for j in X:
        extrastole_MFCC.append(j)
extrastole_labels = [1 for items in extrastole_MFCC]

extrahls_files=glob.glob(extrahls_path+'*.wav')
extrahls_MFCC = []
for i in range(len(extrahls_files)):
    X = extract_and_augment_mfcc_array(
            extrahls_files[i],
            sr=22050,
            duration=10.0,
            n_mfcc=50,
            noise_factor=0.01,
            stretch_rate=1.2,
            shift_max=0.2)
    for j in X:
        extrahls_MFCC.append(j)
extrahls_labels = [2 for items in extrahls_MFCC]

murmur_files=glob.glob(murmur_path+'*.wav')
murmur_MFCC = []
for i in range(len(murmur_files)):
    X = extract_and_augment_mfcc_array(
            murmur_files[i],
            sr=22050,
            duration=10.0,
            n_mfcc=50,
            noise_factor=0.01,
            stretch_rate=1.2,
            shift_max=0.2)
    for j in X:
        murmur_MFCC.append(j)
murmur_labels = [3 for items in murmur_MFCC]

artifact_files=glob.glob(artifact_path+'*.wav')
artifact_MFCC = []
for i in range(len(artifact_files)):
    X = extract_and_augment_mfcc_array(
            artifact_files[i],
            sr=22050,
            duration=10.0,
            n_mfcc=50,
            noise_factor=0.01,
            stretch_rate=1.2,
            shift_max=0.2)
    for j in X:
        artifact_MFCC.append(j)
artifact_labels = [4 for items in artifact_MFCC]


#%% 
# 8

#combining all data into input and output arrays
x_data = np.concatenate((normal_MFCC, extrastole_MFCC,extrahls_MFCC,murmur_MFCC,artifact_MFCC))
y_data = np.concatenate((normal_labels, extrastole_labels,extrahls_labels,murmur_labels,artifact_labels))


#%% 
# 9

#splitting arrays into training, validation, test data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=12, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=12, shuffle=True)

# One-Hot encoding for classes
num_classes = len(CLASSES)
eye = np.eye(num_classes, dtype=np.float32)

y_train = eye[y_train]
y_test  = eye[y_test]
y_val   = eye[y_val]


#%% 
# 10

#balancing the dataset by adjusting weights
TRAIN_IMG_COUNT = 585
COUNT_0 = len(normal_labels)/4
COUNT_1 = len(extrastole_labels)/4
COUNT_2 = len(extrahls_labels)/4
COUNT_3 = len(murmur_labels)/4
COUNT_4 = len(artifact_labels)/4

weight_for_0 = TRAIN_IMG_COUNT / (4 * COUNT_0)
weight_for_1 = TRAIN_IMG_COUNT / (4 * COUNT_1)
weight_for_2 = TRAIN_IMG_COUNT / (4 * COUNT_2)
weight_for_3 = TRAIN_IMG_COUNT / (4 * COUNT_3)
weight_for_4 = TRAIN_IMG_COUNT / (4 * COUNT_4)

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3, 4: weight_for_4}

#%% 
# 11

class CNNLSTMHeartbeat(nn.Module):
    def __init__(self, n_mfcc=50, num_classes=5):
        super(CNNLSTMHeartbeat, self).__init__()

        # Input shape: (batch, seq_len, n_mfcc)
        # After permute: (batch, n_mfcc, seq_len)

        self.conv1 = nn.Conv1d(in_channels=n_mfcc, out_channels=2048, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(2048)

        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(1024)

        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)

        # LSTM input_size must match conv output channels
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, n_mfcc) → (batch, n_mfcc, seq_len)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)

        # Now: x is (batch, channels=512, seq_len)
        x = x.permute(0, 2, 1)  # → (batch, seq_len, features) for LSTM

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]  # last time step

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


#%% 
# 12

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

# Reshape input: (batch_size, sequence_length, input_dim)
#x_train_tensor = x_train_tensor.view(x_train_tensor.shape[0], -1, 1)
#_val_tensor = x_val_tensor.view(x_val_tensor.shape[0], -1, 1)
#x_test_tensor = x_test_tensor.view(x_test_tensor.shape[0], -1, 1)

#%% 
# 13

# Create DataLoader
batch_size = 32

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#%% 
# 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMHeartbeat().to(device)

criterion = nn.NLLLoss(weight=torch.tensor(
    [class_weight[i] for i in range(5)],
    dtype=torch.float32
).to(device))

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            val_loss += criterion(outputs, yb).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / len(val_dataset))

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {correct / len(val_dataset):.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()

#%% 
# 15

# model evaluation
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        _, preds = torch.max(outputs, 1)
        correct += (preds == yb).sum().item()

print(f"Test Accuracy: {correct / len(test_dataset):.4f}")

# %%
torch.save(model.state_dict(), 'HeartbeatCNNLSTMmodel.pth')

# %%
torch.load('HeartbeatCNNLSTMmodel.pth', weights_only=False)
model.eval()