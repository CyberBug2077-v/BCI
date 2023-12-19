import torch
import torch.nn as nn
import torch.optim as optim
from discriminator import DiscriminatorT, DiscriminatorF
from module import Generator
import preprocessing

ppg_data = ...
ecg_data = ...
sampling_rate = 1000

dataloader = preprocessing.get_dataloader(ppg_data, ecg_data, sampling_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input size and parameters
input_channels = 1
signal_length = 512  # Time domain signal length
spectrogram_length = 128  # Frequency domain signal length

# Initialize the model
generator = Generator(input_channels, input_channels, [64, 128, 256, 512], [3, 3, 3, 3]).to(device)
discriminatorT = DiscriminatorT(input_channels, signal_length).to(device)
discriminatorF = DiscriminatorF(input_channels, spectrogram_length).to(device)

# Optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_DT = optim.Adam(discriminatorT.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_DF = optim.Adam(discriminatorF.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Binary cross entropy loss
criterion = nn.BCELoss()

# Train
num_epochs = 10  # train round
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        # Real data and label
        real_data = data['ecg'].to(device)
        real_labels = torch.ones(real_data.size(0), 1).to(device)
        fake_labels = torch.zeros(real_data.size(0), 1).to(device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generator fake data
        fake_data = generator(data['ppg'].to(device))

        # Compute loss
        loss_G = criterion(discriminatorT(fake_data), real_labels) + criterion(discriminatorF(fake_data), real_labels)
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train time domain discriminator
        # ---------------------
        optimizer_DT.zero_grad()

        # Compute time
        loss_DT_real = criterion(discriminatorT(real_data), real_labels)
        loss_DT_fake = criterion(discriminatorT(fake_data.detach()), fake_labels)
        loss_DT = (loss_DT_real + loss_DT_fake) / 2
        loss_DT.backward()
        optimizer_DT.step()

        # ---------------------
        #  Train frequency domain discriminator
        # ---------------------
        optimizer_DF.zero_grad()

        # Compute frequency domain discrimination loss
        # Convert time serial data to frequency data
        real_data_freq = preprocessing.transform_to_frequency(real_data)
        fake_data_freq = preprocessing.transform_to_frequency(fake_data.detach())

        loss_DF_real = criterion(discriminatorF(real_data_freq), real_labels)
        loss_DF_fake = criterion(discriminatorF(fake_data_freq), fake_labels)
        loss_DF = (loss_DF_real + loss_DF_fake) / 2
        loss_DF.backward()
        optimizer_DF.step()

        # Display train loss
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                  Loss G: {loss_G.item()}, Loss DT: {loss_DT.item()}, Loss DF: {loss_DF.item()}")
