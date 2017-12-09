import pandas as pd
import matplotlib.pyplot as plt

def plot_losses_from_file(losses_file):
    losses = pd.read_csv(losses_file, header=None)
    enc_dec_loss = losses.iloc[:, 2] + losses.iloc[:, 3]
    disc_loss = losses.iloc[:, 5] + losses.iloc[:, 6]
    plt.plot(range(losses.shape[0]), enc_dec_loss, label='Encoder + Decoder Loss')
    plt.plot(range(losses.shape[0]), disc_loss, label='Discriminator Loss')
    plt.legend()
    plt.title('Encoder + Decoder Loss vs. Discriminator Loss')
    plt.xlabel('# Batches Trained')
    plt.ylabel('Loss')
    plt.xlim([0, losses.shape[0]])
    plt.show()

plot_losses_from_file('~/Desktop/model_losses_8.csv')