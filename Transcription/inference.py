import torch
import torchaudio
import os
import pandas as pd

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata
meta_data_path = r"your metadata path"
try:
    meta_data = pd.read_csv(meta_data_path, sep="|", header=None, quoting=3)
    meta_data.columns = ['file_name', 'transcription', 'normalized_transcription']
    meta_data = meta_data[['file_name', 'normalized_transcription']]
except FileNotFoundError:
    print(f"Metadata file not found at {meta_data_path}")
    exit(1)

# Character mapping (must match training)
characters = list("abcdefghijklmnopqrstuvwxyz'?! ")
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
char_to_num[''] = 0  # CTC blank
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Audio preprocessing parameters (must match training)
frame_length, frame_step, fft_length = 256, 160, 384

# Model definition (must match training)
def get_model(input_dim, output_dim, rnn_layers=5, rnn_units=256):
    class ASRModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, (11, 41), (2, 2), (5, 20), bias=False)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.conv2 = torch.nn.Conv2d(32, 32, (11, 21), (1, 2), (5, 10), bias=False)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.reshape = lambda x: x.view(x.size(0), x.size(2), -1)
            # Dummy forward to infer GRU input size dynamically
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, input_dim, 100)  # (B, C=1, FREQ, TIME)
                x = torch.relu(self.bn1(self.conv1(dummy_input)))
                x = torch.relu(self.bn2(self.conv2(x)))
                b, c, f, t = x.shape
                gru_input_size = c * f

            self.gru = torch.nn.ModuleList([
                torch.nn.GRU(
                    input_size=gru_input_size if i == 0 else rnn_units * 2,
                    hidden_size=rnn_units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                ) for i in range(rnn_layers)
            ])
            self.dropout = torch.nn.Dropout(0.5)
            self.fc1 = torch.nn.Linear(rnn_units * 2, rnn_units * 2)
            self.fc2 = torch.nn.Linear(rnn_units * 2, output_dim + 1)

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

# Decoder (same as training)
def decode_prediction(pred, lengths):
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
    return decoded[0]  # Single audio input

# Process audio input
def process_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.lower().endswith('.wav'):
        raise ValueError("Audio file must be in WAV format")
    audio, _ = torchaudio.load(audio_path)
    audio = audio.squeeze(0)  # Remove channel dimension
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=fft_length, win_length=frame_length, hop_length=frame_step, power=2)(audio)
    spectrogram = torch.sqrt(spectrogram + 1e-10)
    spectrogram = (spectrogram - spectrogram.mean(dim=1, keepdim=True)) / (spectrogram.std(dim=1, keepdim=True) + 1e-10)
    spectrogram = spectrogram.transpose(0, 1)  # TIME x FREQ
    spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension (1, T, F)
    spectrogram = spectrogram.permute(0, 2, 1)  # (1, F, T)
    return spectrogram, torch.tensor([spectrogram.shape[2]])  # Spectrogram and its length

# Get target transcription from metadata
def get_target_transcription(audio_path):
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    target_row = meta_data[meta_data['file_name'] == file_name]
    if not target_row.empty:
        return target_row.iloc[0]['normalized_transcription'].lower()
    return None

# Inference function
def infer(model, audio_path):
    model.eval()
    with torch.no_grad():
        spectrogram, spec_length = process_audio(audio_path)
        spectrogram = spectrogram.to(device)
        spec_length = spec_length.to(device)
        outputs, out_lengths = model(spectrogram, spec_length)
        transcription = decode_prediction(outputs, out_lengths)
    target_transcription = get_target_transcription(audio_path)
    return transcription, target_transcription

# Main execution
if __name__ == "__main__":
    # Load model
    model_path = r"your saved model.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        exit(1)
    model = get_model(input_dim=fft_length // 2 + 1, output_dim=len(characters)).to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Interactive loop
    while True:
        audio_path = input("Enter the path to the WAV audio file (or 'exit' to quit): ").strip()
        if audio_path.lower() == 'exit':
            print("Exiting...")
            break
        try:
            transcription, target_transcription = infer(model, audio_path)
            print(f"Predicted Transcription: {transcription}")
            if target_transcription:
                print(f"Target Transcription: {target_transcription}")
            else:
                print("Target Transcription: Not found in metadata")
        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Error processing audio: {e}")
        