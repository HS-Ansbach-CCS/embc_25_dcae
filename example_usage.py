#!/usr/bin/env python3
"""
Example script for training and evaluating the EEG autoencoder

author: Annika Stiehl
date: 2025-07-12
This script demonstrates how to use the DeepConvolutionalAutoencoder
class for training on EEG data it is not a complete implementation
of the paper, but provides a good starting point for experimentation.
"""

from model import DeepConvolutionalAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# plotting setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_eeg_data(parquet_file):
    """Load EEG data from parquet file"""
    print(f"Loading data from: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # get channel names (exclude metadata columns)
    channel_names = [col for col in df.columns if col not in [
        'window_label', 'window_id']]
    print(f"Found {len(channel_names)} EEG channels")

    window_ids = df['window_id'].unique()
    print(f"Number of windows: {len(window_ids)}")
    print(
        f"Labels: {df['window_label'].value_counts().sort_index().to_dict()}")

    # extract data by window
    data_windows = []
    window_labels = []
    window_id_list = []

    for window_id in window_ids:
        window_data = df[df['window_id'] == window_id]

        # get EEG data - shape: (time_steps, channels) -> (channels, time_steps)
        eeg_data = window_data[channel_names].values.T
        label = window_data['window_label'].iloc[0]

        data_windows.append(eeg_data)
        window_labels.append(label)
        window_id_list.append(window_id)

    data = np.array(data_windows)  # (n_windows, channels, time_steps)
    labels = np.array(window_labels)
    window_ids = np.array(window_id_list)

    print(f"Final data shape: {data.shape}")
    return data, labels, window_ids, channel_names


def create_model(input_channels, time_steps, latent_dim=500):
    """Create the autoencoder model"""
    print(
        f"Creating model: {input_channels} channels, {time_steps} time steps, latent_dim={latent_dim}")

    model = DeepConvolutionalAutoencoder(
        time_steps=time_steps, latent_dim=latent_dim)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    return model


def train_model(model, train_data, val_data, num_epochs=50, lr=0.001, batch_size=8):
    """Train the autoencoder"""
    print(f"Training for {num_epochs} epochs...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # convert to tensors
    train_tensor = torch.FloatTensor(train_data).to(device)
    val_tensor = torch.FloatTensor(val_data).to(device)

    # setup training
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # track training
    train_losses = []
    val_losses = []
    learning_rates = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []

        # mini-batch training
        for i in range(0, len(train_tensor), batch_size):
            batch_data = train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            reconstructed, latent = model(batch_data)

            # handle shape issues
            if len(reconstructed.shape) == 4 and len(batch_data.shape) == 3:
                batch_data = batch_data.unsqueeze(1)

            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            val_reconstructed, _ = model(val_tensor)

            val_target = val_tensor
            if len(val_reconstructed.shape) == 4 and len(val_target.shape) == 3:
                val_target = val_target.unsqueeze(1)

            val_loss = criterion(val_reconstructed, val_target).item()

        # save metrics
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f'Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}')

    print(f"Training done!")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rate': learning_rates
    }
    return history


def evaluate_model(model, test_data, test_labels, test_ids, channel_names):
    """Evaluate model performance"""
    print("Evaluating model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_tensor = torch.FloatTensor(test_data).to(device)

    with torch.no_grad():
        reconstructed, latent = model(test_tensor)

        # fix shape if needed
        if len(reconstructed.shape) == 4:
            reconstructed = reconstructed.squeeze(1)

    # convert back to numpy
    original = test_data
    reconstructed = reconstructed.cpu().numpy()
    latent = latent.cpu().numpy()

    # calculate overall metrics
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())

    # per-channel metrics
    channel_mse = []
    for ch in range(original.shape[1]):
        ch_mse = mean_squared_error(
            original[:, ch, :].flatten(),
            reconstructed[:, ch, :].flatten()
        )
        channel_mse.append(ch_mse)

    # per-sample metrics
    sample_mse = []
    for i in range(len(original)):
        sample_mse.append(mean_squared_error(
            original[i].flatten(), reconstructed[i].flatten()
        ))

    results = {
        'original': original,
        'reconstructed': reconstructed,
        'latent': latent,
        'mse': mse,
        'mae': mae,
        'channel_mse': channel_mse,
        'sample_mse': sample_mse,
        'test_labels': test_labels,
        'test_ids': test_ids
    }

    print(f"Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(
        f"  Avg sample MSE: {np.mean(sample_mse):.6f} ± {np.std(sample_mse):.6f}")

    return results


def plot_training_history(history, save_fig=False, save_path='figures/training_history.png'):
    """Plot training curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # plot losses
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if save_fig:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to: {save_path}")


def plot_reconstruction_examples(results, channel_names, num_examples=3,
                                 channels_to_plot=None, save_fig=False,
                                 save_path='figures/reconstruction_examples.png'):
    """Plot some reconstruction examples"""
    original = results['original']
    reconstructed = results['reconstructed']
    test_labels = results['test_labels']
    test_ids = results['test_ids']

    if channels_to_plot is None:
        channels_to_plot = [0, 1, 2, 3]  # first 4 channels

    num_channels = len(channels_to_plot)

    # select examples (try to get mix of labels)
    unique_labels = np.unique(test_labels)
    example_indices = []

    for label in unique_labels:
        label_indices = np.where(test_labels == label)[0]
        example_indices.extend(
            label_indices[:num_examples//len(unique_labels) + 1])

    example_indices = example_indices[:num_examples]

    fig, axes = plt.subplots(num_examples, num_channels,
                             figsize=(20, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    if num_channels == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(example_indices):
        for j, ch_idx in enumerate(channels_to_plot):
            ax = axes[i, j]

            time_steps = original.shape[2]
            time_vector = np.arange(time_steps)

            # plot original vs reconstructed
            ax.plot(time_vector, original[idx, ch_idx, :],
                    label='Original', linewidth=1.5, alpha=0.8)
            ax.plot(time_vector, reconstructed[idx, ch_idx, :],
                    label='Reconstructed', linewidth=1.5, alpha=0.8)

            # calculate error for this channel
            ch_mae = mean_absolute_error(original[idx, ch_idx, :],
                                         reconstructed[idx, ch_idx, :])

            ax.set_title(f'Window {test_ids[idx]}, Label {test_labels[idx]}\n'
                         f'{channel_names[ch_idx]}, MAE: {ch_mae:.4f}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Amplitude (μV)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if save_fig:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to: {save_path}")


def plot_reconstruction_errors(results, channel_names, num_examples=2, save_fig=False, save_path='figures/reconstruction_errors.png'):
    """Plot reconstruction errors for selected examples."""
    original = results['original']
    reconstructed = results['reconstructed']
    test_labels = results['test_labels']
    test_ids = results['test_ids']

    # Select examples
    example_indices = [0, len(original)//2] if len(original) > 1 else [0]
    example_indices = example_indices[:num_examples]

    fig, axes = plt.subplots(num_examples, 2, figsize=(16, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(example_indices):
        # Absolute error over time for all channels
        errors = np.abs(original[idx] - reconstructed[idx])

        # Plot 1: Error heatmap across channels and time
        im = axes[i, 0].imshow(errors, aspect='auto',
                               cmap='Reds', interpolation='nearest')
        axes[i, 0].set_title(
            f'Reconstruction Error Heatmap\nWindow ID: {test_ids[idx]}, Label: {test_labels[idx]}')
        axes[i, 0].set_xlabel('Time Steps')
        axes[i, 0].set_ylabel('Channel Index')
        axes[i, 0].set_yticks(
            range(0, len(channel_names), max(1, len(channel_names)//10)))
        axes[i, 0].set_yticklabels([channel_names[j] for j in range(
            0, len(channel_names), max(1, len(channel_names)//10))])
        plt.colorbar(im, ax=axes[i, 0], label='Absolute Error')

        # Plot 2: Average error per channel
        channel_errors = np.mean(errors, axis=1)
        bars = axes[i, 1].bar(range(len(channel_names)), channel_errors)
        axes[i, 1].set_title(f'Average Reconstruction Error per Channel')
        axes[i, 1].set_xlabel('Channel')
        axes[i, 1].set_ylabel('Mean Absolute Error')
        axes[i, 1].set_xticks(
            range(0, len(channel_names), max(1, len(channel_names)//10)))
        axes[i, 1].set_xticklabels([channel_names[j] for j in range(
            0, len(channel_names), max(1, len(channel_names)//10))], rotation=45)
        axes[i, 1].grid(True, alpha=0.3)

        # Color bars by error level
        max_error = np.max(channel_errors)
        for bar, error in zip(bars, channel_errors):
            bar.set_color(plt.cm.Reds(error / max_error))

    plt.tight_layout()
    plt.show()

    if save_fig:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")


def preprocess_data(data, normalize=True):
    """Preprocess EEG data"""
    print("Preprocessing data...")

    if normalize:
        # In the paper is mentioned to normalize with a 5% / 95 % # quantile normalization (from all the training data channel dependent)
        # each channel independently
        # in the example a standard scaler is used
        orig_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])  # flatten for scaling
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_flat)
        data = data_normalized.reshape(orig_shape)
        print("Data normalized using StandardScaler")
        return data, scaler
    else:
        return data, None


def main():
    """Main function"""
    print("="*50)
    print("EEG Autoencoder Training Script")
    print("="*50)

    # config
    config = {
        'parquet_file': 'tuh_data_dev_preprocessed/windows_1_balanced.parquet',
        'latent_dim': 500,
        'epochs': 50,
        'lr': 0.001,
        'batch_size': 8,
        'test_size': 0.3,
        'random_state': 42
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # set seeds
    torch.manual_seed(config['random_state'])
    np.random.seed(config['random_state'])

    # load data
    data, labels, window_ids, channel_names = load_eeg_data(
        config['parquet_file'])

    # preprocess
    data_preprocessed, scaler = preprocess_data(data, normalize=True)

    # train/test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        data_preprocessed, labels, window_ids,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=labels
    )

    print(f"Data split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Train labels: {np.bincount(y_train)}")
    print(f"  Test labels: {np.bincount(y_test)}")
    print()

    # create model
    model = create_model(
        input_channels=data.shape[1],
        time_steps=data.shape[2],
        latent_dim=config['latent_dim']
    )

    # train
    print("Starting training...")
    history = train_model(
        model, X_train, X_test,
        num_epochs=config['epochs'],
        lr=config['lr'],
        batch_size=config['batch_size']
    )

    # evaluate
    print("Evaluating...")
    results_train = evaluate_model(
        model, X_train, y_train, ids_train, channel_names)
    results_test = evaluate_model(
        model, X_test, y_test, ids_test, channel_names)

    # visualize
    print("Creating plots...")
    plot_training_history(history, save_fig=True)
    plot_reconstruction_examples(results_test, channel_names, num_examples=2,
                                 channels_to_plot=[0, 1, 10, 15], save_fig=True)
    plot_reconstruction_errors(
        results_test, channel_names, num_examples=2, save_fig=True)

    print("Done!")


if __name__ == "__main__":
    main()
