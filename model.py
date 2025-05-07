import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTMModel, self).__init__()

        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout_cnn = nn.Dropout(0.5) 

        # LSTM layers
        self.lstm = nn.LSTM(input_size=512*8*8, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True) # Bidirectional LSTM

        # Fully connected layer
        self.fc = nn.Linear(512*2, num_classes)  # LSTM bidirectional olduğu için *2

    def forward(self, x, lengths):
        batch_size, timesteps, C, H, W = x.size()

        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn_layers(c_in)
        c_out = self.dropout_cnn(c_out) 

        r_in = c_out.view(batch_size, timesteps, -1)

        # Pack padded sequence
        packed_input = pack_padded_sequence(r_in, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM layers
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # Unpack padded sequence
        r_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = self.fc(r_out[:, -1, :]) 
        return output