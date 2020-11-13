import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader



class Generator(nn.Module):
    """Generator Class

    Args:
        input_dim (int) : the dimension of the input vector
        im_chan (int) : the number of channels in the images, fitted for the dataset used
        hidden_dim (int) : the inner dimension
    """
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            self.get_gen_block(input_dim, hidden_dim*4),
            self.get_gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self.get_gen_block(hidden_dim*2, hidden_dim),
            self.get_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def get_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """Function to return a sequence of operations corresponding to a generator block of DCGAN;
            a transposed convolution, a batchnorm (except in the final layer), and an activation.

        Args:
            input_channels (int) : how many channels the input feature representation has
            output_channels (int) : how many channels the output feature representation should have
            kernel_size (int, optional): the size of each convolutional filter (size, size). Defaults to 3.
            stride (int, optional): the stride of the convolution. Defaults to 2.
            final_layer (bool, optional): True if last layer, False otherwise. Defaults to False.
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
                )

    def forward(self, noise):
        """Function for completing a forward pass of the generator: Given a noise tensor, returns generated images

        Args:
            noise (tensor): a noise tensor with dimensions (n_samples, input_dim)
        """
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


def get_noise(n_samples, input_dim, device="cpu"):
    """Function for creating noise vector from normal distribution: Given the dimensions (n_samples, input_dim)

    Args:
        n_samples (int): the number of samples to generate
        input_dim (int): the dimension of the input vector
        device (str, optional): the device type. Defaults to "cpu".
    """
    return torch.randn(n_samples, input_dim, device=device)


class Discriminator(nn.Module):
    """Discriminator

    Args:
        im_chan (int): the number of channels in the images, fitted for the dataset used
        hidden_dim (int): the inner dimension
    """
    def __init__(self, im_chan, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_disc_block(im_chan, hidden_dim),
            self.get_disc_block(hidden_dim, hidden_dim*2),
            self.get_disc_block(hidden_dim*2, 1, final_layer=True),
        )
        
    def get_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """Function to return a sequence of operations corresponding to a discriminator block of the DCGAN;
           a convolution, a batchnorm (not final layer), and an activation (not final layer)

        Args:
            input_channels (int): how many channels the input feature representation has
            output_channels (int): how many channels the output feature representation should have
            kernel_size (int): the size of each convolutional filter (size, size)
            stride (int): the stride of the convolution
            final_layer (bool, optional): True if final layer, otherwise False. Defaults to False.
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        """Method for completing a forward pass of the discriminator: Given an image tensor,
           returns a 1-dimensional tensor representing fake/real.

        Args:
            image (tensor): a flattened image tensor with dimension (im_chan)
        """
        disc_pred = self.disc(image)
        print(f"\n\n\nType of disc pred: {type(disc_pred)}\n\n\n")
        return disc_pred.view(len(disc_pred), -1)


def get_one_hot_labels(labels, n_classes):
    """Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes)

    Args:
        labels (tensor): tensor of labels from the dataloader, size ?
        n_classes (int): the total number of classes in the dataase
    """
    return F.one_hot(labels, n_classes)


def combine_vectors(x, y):
    """Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).

    Args:
        x (tensor): the first vector
        y (tensor): the second vector
    """
    try :
        return torch.cat((x.float(), y.float()), dim=1)
    except Exception as e:
        print(f"[ERROR]: {e} \t x: {x.size()} and y: {y.size()}")
        return None

    
def get_input_dimensions(z_dim, image_shape, n_classes):
    """Function for getting the size of the conditional input dimensions

    Args:
        z_dim (int) : the dimension of the noise vector.
        image_shape (tuple int): the shape of each image as (C, W, H). For MNIST is (1, 28, 28)
        n_classes (int) : the total number of classes in the dataset. For MNIST is 10

    Returns:
        generator_input_dim (int) : the input dimensionality of the conditional generator, which takes the noise and class vectors
        discriminator_im_chan (int) : the number of inputs of input channels to the discriminator
    """
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = image_shape[0] + n_classes

    return generator_input_dim, discriminator_im_chan


def weights_init(m):
    """
    Function to initialize normalized weights and bias for the neural networks
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.bias, 0)
    