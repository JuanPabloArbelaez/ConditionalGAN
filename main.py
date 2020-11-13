from conditional_gan import *
from visualize import *


# Training parameters
IMAGE_SHAPE = (1, 28, 28)
N_CLASSES = 10
CRITERION = nn.BCEWithLogitsLoss()
N_EPOCHS = 200
Z_DIM = 64
DISPLAY_STEP = 500
BATCH_SIZE = 128
LR = 0.0002
DEVICE = "cuda"

# Load Data
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Create Generator & Discriminator objects
generator_input_dim, discriminator_im_chan = get_input_dimensions(Z_DIM, IMAGE_SHAPE, N_CLASSES)
gen = Generator(generator_input_dim).to(DEVICE)
gen_opt = torch.optim.Adam(params=gen.parameters(), lr=LR)

# Create optimizers
disc = Discriminator(discriminator_im_chan).to(DEVICE)
disc_opt = torch.optim.Adam(params=disc.parameters(), lr=LR)

# Generate Weights and Bias
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


def train_neural_networks():
    cur_step = 0
    generator_losses = []
    discriminator_losses = []

    # noise_and_labels = False
    # fake = False

    # fake_image_and_labels = False
    # real_image_and_labels = False
    # disc_fake_pred = False
    # disc_real_pred = False

    for epoch in range(N_EPOCHS):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(DEVICE)
            one_hot_labels = get_one_hot_labels(labels.to(DEVICE), N_CLASSES)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, IMAGE_SHAPE[1], IMAGE_SHAPE[2])

            ### Update discriminator ###
            # zero out the gradients
            disc_opt.zero_grad()
            # get noise corresponding to the curren batch_size
            noise = get_noise(cur_batch_size, Z_DIM, DEVICE)
            # combine noise and labels
            noise_and_labels = combine_vectors(noise, one_hot_labels)
            # generate fake image with generator
            fake = gen(noise_and_labels)

            # combine fake image(detached) with image-one-hot-vector
            fake_image_and_labels = combine_vectors(fake.detach(), image_one_hot_labels)
            # combine real image with image-one-hot-vector
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)

            # discriminator predictions
            fake_disc_pred = disc(fake_image_and_labels)
            real_disc_pred = disc(real_image_and_labels)

            # loss
            fake_disc_loss = CRITERION(fake_disc_pred, torch.zeros_like(fake_disc_pred))
            real_disc_loss = CRITERION(real_disc_pred, torch.ones_like(real_disc_pred))
            disc_loss = (fake_disc_loss + real_disc_loss) / 2
            

            # Get gradient & update parameters
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Keep track of average discriminator loss
            discriminator_losses += [disc_loss.item()]

            ### Update Generator ###
            # zero gradients
            gen_opt.zero_grad()

            # combine fake image and image-one-hot-labels
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
            # get discriminator prediction 
            fake_disc_pred = disc(fake_image_and_labels)
            # generator loss
            gen_loss = CRITERION(fake_disc_pred, torch.ones_like(fake_disc_pred))

            # get gradients and update parameters
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            # keep track of the generator loss
            generator_losses += [gen_loss.item()]

            
            ## Visualize ##
            if cur_step % DISPLAY_STEP == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                disc_mean = sum(discriminator_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                print(f"Epoch: {epoch}   Step: {cur_step}   Gen Loss: {gen_mean}   Disc Loss: {disc_mean}")
                
                # Show
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Discrominator Loss"
                )
                plt.legend()
                plt.show()
            
            cur_step += 1


if __name__ == "__main__":
    train_neural_networks()