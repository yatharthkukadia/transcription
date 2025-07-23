import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from jiwer import wer
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


audio_dir = r"your audio directory"
meta_data = pd.read_csv(r"your metadata.csv", sep="|", header=None, quoting=3)
meta_data.columns = ['file_name', 'transcription', 'normalized_transcription']
meta_data = meta_data[['file_name', 'normalized_transcription']].sample(frac=1).reset_index(drop=True)


split = int(len(meta_data) * 0.90)
train_data, test_data = meta_data[:split], meta_data[split:]


characters = list("abcdefghijklmnopqrstuvwxyz'?! ")
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
char_to_num[''] = 0  # CTC blank
num_to_char = {idx: char for char, idx in char_to_num.items()}


frame_length, frame_step, fft_length = 256, 160, 384

class ASRDataset(Dataset):
    def __init__(self, data, audio_dir):
        self.data = data
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_file = self.data.iloc[idx]['file_name']
        label = self.data.iloc[idx]['normalized_transcription'].lower()
        audio, _ = torchaudio.load(os.path.join(self.audio_dir, wav_file + '.wav'))
        audio = audio.squeeze(0)
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=fft_length, win_length=frame_length, hop_length=frame_step, power=2)(audio)
        spectrogram = torch.sqrt(spectrogram + 1e-10)
        spectrogram = (spectrogram - spectrogram.mean(dim=1, keepdim=True)) / (spectrogram.std(dim=1, keepdim=True) + 1e-10)
        spectrogram = spectrogram.transpose(0, 1)  # TIME x FREQ
        label = [char_to_num.get(c, 0) for c in label]
        return spectrogram, torch.tensor(label, dtype=torch.int64)

def collate_fn(batch):
    spectrograms, labels = zip(*batch)
    spectrogram_lengths = torch.tensor([s.shape[0] for s in spectrograms])
    label_lengths = torch.tensor([len(l) for l in labels])
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)  # (B, T, F)
    spectrograms = spectrograms.permute(0, 2, 1)  # (B, F, T)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return spectrograms, labels, spectrogram_lengths, label_lengths

# DataLoaders
batch_size = 16
train_loader = DataLoader(ASRDataset(train_data, audio_dir), batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
test_loader = DataLoader(ASRDataset(test_data, audio_dir), batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

# Model
def get_model(input_dim, output_dim, rnn_layers=5, rnn_units=256):
    class ASRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, (11, 41), (2, 2), (5, 20), bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, (11, 21), (1, 2), (5, 10), bias=False)
            self.bn2 = nn.BatchNorm2d(32)
            self.reshape = lambda x: x.view(x.size(0), x.size(2), -1)
            # Dummy forward to infer GRU input size dynamically
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, input_dim, 100)  # (B, C=1, FREQ, TIME)
                x = torch.relu(self.bn1(self.conv1(dummy_input)))
                x = torch.relu(self.bn2(self.conv2(x)))
                b, c, f, t = x.shape
                gru_input_size = c * f

            self.gru = nn.ModuleList([
                nn.GRU(
                    input_size=gru_input_size if i == 0 else rnn_units * 2,
                    hidden_size=rnn_units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )for i in range(rnn_layers)
            ])
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(rnn_units * 2, rnn_units * 2)
            self.fc2 = nn.Linear(rnn_units * 2, output_dim + 1)

        def forward(self, x, lengths):
            x = x.unsqueeze(1)
            x = self.dropout(torch.relu(self.bn1(self.conv1(x))))
            x = self.dropout(torch.relu(self.bn2(self.conv2(x))))
            x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
            x = x.reshape(x.size(0), x.size(1), -1)  # (B, T, C*F)
            for i, gru in enumerate(self.gru):
                x, _ = gru(x)
                if i < len(self.gru) - 1:
                    x = self.dropout(x)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            return torch.log_softmax(x, dim=-1), lengths // 4

    return ASRModel()

model = get_model(input_dim=fft_length // 2 + 1, output_dim=len(characters)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# Decoder
def decode_batch_prediction(pred, lengths):
    pred = pred.cpu()  # (B, T, C)
    decoded = []
    for i, p in enumerate(pred):  # Loop over batch
        p = p[:lengths[i]]  # Only keep valid time steps
        argmax = p.argmax(dim=-1)
        decoded_indices, prev = [], -1
        for idx in argmax:
            if idx.item() != prev and idx.item() != 0:
                decoded_indices.append(idx.item())
            prev = idx.item()
        decoded.append(''.join(num_to_char.get(i, '') for i in decoded_indices))
    return decoded

class CallBackEval:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def on_epoch_end(self, epoch):
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for spectrograms, labels, spec_lengths, label_lengths in self.dataloader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                spec_lengths = spec_lengths.to(device)
                outputs, out_lengths = model(spectrograms, spec_lengths)
                predictions.extend(decode_batch_prediction(outputs, out_lengths))
                for label in labels:
                    text = ''.join(num_to_char.get(idx.item(), '') for idx in label if idx != 0)
                    targets.append(text)
        print("-" * 80)
        print(f"Epoch {epoch + 1} - WER: {wer(targets, predictions):.4f}")
        print("-" * 80)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target: {targets[i]}\nPrediction: {predictions[i]}\n{'-'*80}")

# Training loop
epochs = 30
callback = CallBackEval(test_loader)
for epoch in range(epochs):
    model.train()
    for batch_idx, (spectrograms, labels, spec_lengths, label_lengths) in enumerate(train_loader):
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        spec_lengths, label_lengths = spec_lengths.to(device), label_lengths.to(device)
        optimizer.zero_grad()
        try:
            outputs, out_lengths = model(spectrograms, spec_lengths)
            loss = criterion(outputs.permute(1, 0, 2), labels, out_lengths, label_lengths)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            torch.cuda.empty_cache()
            continue
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    callback.on_epoch_end(epoch)

# Save the model
model_save_path = r"save model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")