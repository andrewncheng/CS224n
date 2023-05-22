import torch
import argparse
import matplotlib.pyplot as plt
from VAE import VAE
# configuration

argp = argparse.ArgumentParser()
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--save_figures_path', default=None)
args = argp.parse_args()


model = torch.load(args.reading_params_path)
model.eval()
hidden_dim = 300
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
for i in range(10):

    ##########################
    ### RANDOM SAMPLE
    ##########################

    n_images = 10
    rand_features = torch.randn(n_images, hidden_dim).to(device)
    new_images = model.decoder(rand_features)

    ##########################
    ### VISUALIZATION
    ##########################

    image_width = 28

    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(10, 2.5), sharey=True)
    decoded_images = new_images[:n_images]

    for ax, img in zip(axes, decoded_images):
        curr_img = img.detach().to(torch.device('cpu'))
        ax.imshow(curr_img.view((image_width, image_width)), cmap='binary')

    plt.savefig(args.save_figure_path)