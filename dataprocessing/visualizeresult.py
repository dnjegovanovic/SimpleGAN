import matplotlib.pyplot as plt
import itertools
import numpy as np


def visuzlize_result(all_losses, all_d_vals, epoch_samples):
    fig = plt.figure(figsize=(16, 6))

    ## Plotting the losses
    ax = fig.add_subplot(1, 2, 1)
    g_losses = [item[0] for item in itertools.chain(*all_losses)]
    d_losses = [item[1] / 2.0 for item in itertools.chain(*all_losses)]
    plt.plot(g_losses, label='Generator loss', alpha=0.95)
    plt.plot(d_losses, label='Discriminator loss', alpha=0.95)
    plt.legend(fontsize=20)
    ax.set_xlabel('Iteration', size=15)
    ax.set_ylabel('Loss', size=15)

    epochs = np.arange(1, 101)
    epoch2iter = lambda e: e * len(all_losses[-1])
    epoch_ticks = [1, 20, 40, 60, 80, 100]
    newpos = [epoch2iter(e) for e in epoch_ticks]
    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(epoch_ticks)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    ax2.set_xlabel('Epoch', size=15)
    ax2.set_xlim(ax.get_xlim())
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    ## Plotting the outputs of the discriminator
    ax = fig.add_subplot(1, 2, 2)
    d_vals_real = [item[0] for item in itertools.chain(*all_d_vals)]
    d_vals_fake = [item[1] for item in itertools.chain(*all_d_vals)]
    plt.plot(d_vals_real, alpha=0.75, label=r'Real: $D(\mathbf{x})$')
    plt.plot(d_vals_fake, alpha=0.75, label=r'Fake: $D(G(\mathbf{z}))$')
    plt.legend(fontsize=20)
    ax.set_xlabel('Iteration', size=15)
    ax.set_ylabel('Discriminator output', size=15)

    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(epoch_ticks)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    ax2.set_xlabel('Epoch', size=15)
    ax2.set_xlim(ax.get_xlim())
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig('simple-gan-learning-curve.png')
    plt.show()

    selected_epochs = [1, 2, 4, 10, 50, 100]
    fig = plt.figure(figsize=(10, 14))
    for i, e in enumerate(selected_epochs):
        for j in range(5):
            ax = fig.add_subplot(6, 5, i * 5 + j + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.text(
                    -0.06, 0.5, 'Epoch {}'.format(e),
                    rotation=90, size=18, color='red',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=ax.transAxes)

            image = epoch_samples[e - 1][j]
            ax.imshow(image, cmap='gray_r')

    plt.savefig('simple_gan-samples.png')
    plt.show()
