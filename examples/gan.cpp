// Deep Convolutional GAN (DCGAN) Example
// Trains on MNIST to generate realistic handwritten digits

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "mnist.h"
#include "logging.h"
#include <cstdio>
#include <cmath>
#include <random>

// =============================================================================
// Generator: Transforms random noise into images using ConvTranspose2d
// =============================================================================
class Generator : public Module {
public:
    Linear* fc;
    ConvTranspose2d* deconv1;
    ConvTranspose2d* deconv2;
    ConvTranspose2d* deconv3;
    BatchNorm2d* bn1;
    BatchNorm2d* bn2;

    size_t latent_dim;

    Generator(size_t latent_dim = 100) : latent_dim(latent_dim) {
        // latent -> 4x4 -> 7x7 -> 14x14 -> 28x28
        fc = new Linear(latent_dim, 256 * 4 * 4);
        bn1 = new BatchNorm2d(128);
        bn2 = new BatchNorm2d(64);
        deconv1 = new ConvTranspose2d(256, 128, 3, 2, 1, 1);  // 4x4 -> 7x7
        deconv2 = new ConvTranspose2d(128, 64, 4, 2, 1);       // 7x7 -> 14x14
        deconv3 = new ConvTranspose2d(64, 1, 4, 2, 1);         // 14x14 -> 28x28
    }

    ~Generator() {
        delete fc;
        delete bn1;
        delete bn2;
        delete deconv1;
        delete deconv2;
        delete deconv3;
    }

    TensorPtr forward(const TensorPtr& z) override {
        size_t batch = z->shape[0];

        auto h = fc->forward(z)->relu();
        h = h->reshape({batch, 256, 4, 4});

        h = deconv1->forward(h);
        h = bn1->forward(h)->relu();

        h = deconv2->forward(h);
        h = bn2->forward(h)->relu();

        h = deconv3->forward(h)->sigmoid();  // Output in [0, 1]

        return h;
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& p : fc->parameters()) params.push_back(p);
        for (auto& p : deconv1->parameters()) params.push_back(p);
        for (auto& p : deconv2->parameters()) params.push_back(p);
        for (auto& p : deconv3->parameters()) params.push_back(p);
        for (auto& p : bn1->parameters()) params.push_back(p);
        for (auto& p : bn2->parameters()) params.push_back(p);
        return params;
    }

    void train_mode() {
        bn1->train();
        bn2->train();
    }

    void eval_mode() {
        bn1->eval();
        bn2->eval();
    }

    std::string name() const override { return "Generator"; }
};

// =============================================================================
// Discriminator: Classifies images as real or fake using Conv2d
// =============================================================================
class Discriminator : public Module {
public:
    Conv2d* conv1;
    Conv2d* conv2;
    Conv2d* conv3;
    Flatten* flatten;
    Linear* fc;
    Dropout* dropout;

    Discriminator() {
        // 28x28 -> 14x14 -> 7x7 -> 4x4 -> 1
        conv1 = new Conv2d(1, 64, 4, 2, 1);    // 28x28 -> 14x14
        conv2 = new Conv2d(64, 128, 4, 2, 1);  // 14x14 -> 7x7
        conv3 = new Conv2d(128, 256, 3, 2, 1); // 7x7 -> 4x4
        flatten = new Flatten();
        dropout = new Dropout(0.3f);
        fc = new Linear(256 * 4 * 4, 1);
    }

    ~Discriminator() {
        delete conv1;
        delete conv2;
        delete conv3;
        delete flatten;
        delete dropout;
        delete fc;
    }

    TensorPtr forward(const TensorPtr& x) override {
        auto h = conv1->forward(x);
        h = h->relu();  // LeakyReLU would be better, using ReLU for simplicity
        h = dropout->forward(h);

        h = conv2->forward(h);
        h = h->relu();
        h = dropout->forward(h);

        h = conv3->forward(h);
        h = h->relu();

        h = flatten->forward(h);
        h = fc->forward(h);

        return h->sigmoid();  // Output probability [0, 1]
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& p : conv1->parameters()) params.push_back(p);
        for (auto& p : conv2->parameters()) params.push_back(p);
        for (auto& p : conv3->parameters()) params.push_back(p);
        for (auto& p : fc->parameters()) params.push_back(p);
        return params;
    }

    void train_mode() {
        dropout->train();
    }

    void eval_mode() {
        dropout->eval();
    }

    std::string name() const override { return "Discriminator"; }
};

// =============================================================================
// Helper: Generate random noise
// =============================================================================
static std::mt19937 gan_rng(42);

TensorPtr sample_noise(size_t batch_size, size_t latent_dim) {
    auto z = Tensor::create({batch_size, latent_dim}, true);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : z->data) {
        v = dist(gan_rng);
    }
    return z;
}

// =============================================================================
// Helper: Print generated images as ASCII art
// =============================================================================
void print_samples(const TensorPtr& images, int num_cols = 5) {
    const char* chars = " .:-=+*#%@";
    int num_chars = 10;

    size_t batch = images->shape[0];
    size_t num_to_show = std::min(batch, static_cast<size_t>(num_cols));

    // Print images side by side, skipping rows for aspect ratio
    for (size_t row = 0; row < 28; row += 2) {
        printf("    ");
        for (size_t img = 0; img < num_to_show; img++) {
            for (size_t col = 0; col < 28; col++) {
                size_t idx = img * 28 * 28 + row * 28 + col;
                float val = images->data[idx];
                int char_idx = static_cast<int>(val * (num_chars - 1));
                char_idx = std::max(0, std::min(num_chars - 1, char_idx));
                printf("%c", chars[char_idx]);
            }
            printf("  ");
        }
        printf("\n");
    }
}

// =============================================================================
// Helper: Count parameters
// =============================================================================
size_t count_params(Module* model) {
    size_t total = 0;
    for (auto& p : model->parameters()) {
        total += p->data.size();
    }
    return total;
}

// =============================================================================
// Main Training Loop
// =============================================================================
int main() {
    printf("=======================================================\n");
    printf("Deep Convolutional GAN (DCGAN) - MNIST\n");
    printf("=======================================================\n\n");

    // Hyperparameters
    const size_t latent_dim = 100;
    const size_t batch_size = 64;
    const size_t num_epochs = 20;
    const float lr_g = 0.0002f;
    const float lr_d = 0.0002f;
    const float beta1 = 0.5f;
    const float beta2 = 0.999f;

    printf("Configuration:\n");
    printf("  Latent dimension: %zu\n", latent_dim);
    printf("  Batch size: %zu\n", batch_size);
    printf("  Epochs: %zu\n", num_epochs);
    printf("  Generator LR: %.5f\n", lr_g);
    printf("  Discriminator LR: %.5f\n\n", lr_d);

    // Load MNIST data
    printf("Loading MNIST dataset...\n");
    auto train_data = load_mnist_train("data/");
    printf("  Training samples: %zu\n\n", train_data.num_samples);

    // Create models
    Generator generator(latent_dim);
    Discriminator discriminator;

    printf("Model Parameters:\n");
    printf("  Generator: %zu\n", count_params(&generator));
    printf("  Discriminator: %zu\n\n", count_params(&discriminator));

    // Optimizers (Adam with lower beta1 for GANs)
    Adam opt_g(generator.parameters(), lr_g, beta1, beta2);
    Adam opt_d(discriminator.parameters(), lr_d, beta1, beta2);

    // Loss function
    BCELoss criterion;

    // Training logger
    TrainingLogger logger("logs", "gan");
    logger.set_total_epochs(num_epochs);

    // Data loader
    DataLoader train_loader(train_data, batch_size, true);

    // Fixed noise for visualization (to see progression)
    auto fixed_noise = sample_noise(10, latent_dim);

    printf("Training...\n");
    printf("-------------------------------------------------------\n");

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        logger.new_epoch();
        train_loader.reset();

        float epoch_d_loss = 0.0f;
        float epoch_g_loss = 0.0f;
        float epoch_d_real = 0.0f;
        float epoch_d_fake = 0.0f;
        size_t num_batches = 0;

        generator.train_mode();
        discriminator.train_mode();

        while (train_loader.has_next()) {
            auto [real_images, labels] = train_loader.next_batch();
            size_t current_batch = real_images->shape[0];

            // Reshape to [batch, 1, 28, 28]
            real_images = real_images->reshape({current_batch, 1, 28, 28});

            // Create labels
            auto real_labels = Tensor::create({current_batch, 1}, false);
            auto fake_labels = Tensor::create({current_batch, 1}, false);
            for (size_t i = 0; i < current_batch; i++) {
                real_labels->data[i] = 0.9f;  // Label smoothing
                fake_labels->data[i] = 0.0f;
            }

            // =====================================================
            // Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            // =====================================================
            opt_d.zero_grad();

            // Train on real images
            auto d_real_output = discriminator.forward(real_images);
            auto d_real_loss = criterion(d_real_output, real_labels);

            // Generate fake images
            auto z = sample_noise(current_batch, latent_dim);
            TensorPtr fake_images;
            {
                NoGradGuard no_grad;  // Don't track gradients for generator during D training
                fake_images = generator.forward(z);
            }

            // Train on fake images
            auto d_fake_output = discriminator.forward(fake_images);
            auto d_fake_loss = criterion(d_fake_output, fake_labels);

            // Combined D loss
            auto d_loss = d_real_loss->add(d_fake_loss);
            d_loss->backward();
            opt_d.step();

            // Track D accuracy
            float d_real_mean = 0.0f, d_fake_mean = 0.0f;
            for (size_t i = 0; i < current_batch; i++) {
                d_real_mean += d_real_output->data[i];
                d_fake_mean += d_fake_output->data[i];
            }
            d_real_mean /= current_batch;
            d_fake_mean /= current_batch;

            // =====================================================
            // Train Generator: maximize log(D(G(z)))
            // =====================================================
            opt_g.zero_grad();

            // Generate new fake images (need gradients this time)
            z = sample_noise(current_batch, latent_dim);
            fake_images = generator.forward(z);

            // We want D to think these are real
            auto g_output = discriminator.forward(fake_images);
            auto g_loss = criterion(g_output, real_labels);

            g_loss->backward();
            opt_g.step();

            // Logging
            epoch_d_loss += d_loss->item();
            epoch_g_loss += g_loss->item();
            epoch_d_real += d_real_mean;
            epoch_d_fake += d_fake_mean;
            num_batches++;

            logger.log_batch("d_loss", d_loss->item());
            logger.log_batch("g_loss", g_loss->item());
        }

        epoch_d_loss /= num_batches;
        epoch_g_loss /= num_batches;
        epoch_d_real /= num_batches;
        epoch_d_fake /= num_batches;

        logger.log("d_loss", epoch_d_loss);
        logger.log("g_loss", epoch_g_loss);
        logger.log("d_real", epoch_d_real);
        logger.log("d_fake", epoch_d_fake);
        logger.step();

        printf("Epoch %2zu/%zu  D_loss: %.4f  G_loss: %.4f  D(x): %.3f  D(G(z)): %.3f\n",
               epoch + 1, num_epochs, epoch_d_loss, epoch_g_loss, epoch_d_real, epoch_d_fake);

        // Show generated samples every 5 epochs
        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            printf("\nGenerated samples (epoch %zu):\n", epoch + 1);
            generator.eval_mode();
            TensorPtr samples;
            {
                NoGradGuard no_grad;
                samples = generator.forward(fixed_noise);
            }
            // Reshape for printing: remove channel dimension
            samples = samples->reshape({10, 28, 28});
            print_samples(samples, 5);
            printf("\n");
            generator.train_mode();
        }
    }

    printf("-------------------------------------------------------\n\n");

    // Generate final samples
    printf("=======================================================\n");
    printf("Final Generated Samples:\n");
    printf("=======================================================\n\n");

    generator.eval_mode();

    // Generate a grid of samples
    for (int row = 0; row < 2; row++) {
        auto z = sample_noise(5, latent_dim);
        TensorPtr samples;
        {
            NoGradGuard no_grad;
            samples = generator.forward(z);
        }
        samples = samples->reshape({5, 28, 28});
        print_samples(samples, 5);
        printf("\n");
    }

    printf("=======================================================\n");

    // Demonstrate latent space interpolation
    printf("\nLatent Space Interpolation:\n");
    printf("=======================================================\n\n");

    auto z1 = sample_noise(1, latent_dim);
    auto z2 = sample_noise(1, latent_dim);

    printf("Interpolating between two random points in latent space:\n\n");

    int num_steps = 7;
    std::vector<TensorPtr> interpolations;

    for (int step = 0; step < num_steps; step++) {
        float alpha = static_cast<float>(step) / (num_steps - 1);

        auto z_interp = Tensor::create({1, latent_dim}, false);
        for (size_t i = 0; i < latent_dim; i++) {
            z_interp->data[i] = (1.0f - alpha) * z1->data[i] + alpha * z2->data[i];
        }

        TensorPtr sample;
        {
            NoGradGuard no_grad;
            sample = generator.forward(z_interp);
        }
        interpolations.push_back(sample->reshape({28, 28}));
    }

    // Print interpolations
    const char* chars = " .:-=+*#%@";
    int num_chars = 10;

    for (size_t row = 0; row < 28; row += 2) {
        printf("    ");
        for (int step = 0; step < num_steps; step++) {
            for (size_t col = 0; col < 28; col++) {
                float val = interpolations[step]->data[row * 28 + col];
                int idx = static_cast<int>(val * (num_chars - 1));
                idx = std::max(0, std::min(num_chars - 1, idx));
                printf("%c", chars[idx]);
            }
            printf(" ");
        }
        printf("\n");
    }

    printf("\n=======================================================\n");

    // Generate samples for each class (using mode collapse check)
    printf("\nDiversity Check (10 random samples):\n");
    printf("=======================================================\n\n");

    auto diverse_z = sample_noise(10, latent_dim);
    TensorPtr diverse_samples;
    {
        NoGradGuard no_grad;
        diverse_samples = generator.forward(diverse_z);
    }
    diverse_samples = diverse_samples->reshape({10, 28, 28});
    print_samples(diverse_samples, 5);
    printf("\n");
    // Print second row
    auto second_row = Tensor::create({5, 28, 28}, false);
    for (size_t i = 0; i < 5 * 28 * 28; i++) {
        second_row->data[i] = diverse_samples->data[5 * 28 * 28 + i];
    }
    print_samples(second_row, 5);

    printf("\n=======================================================\n");
    printf("Training complete!\n");
    printf("=======================================================\n");

    // Save logs
    logger.save_csv();
    logger.save_json();

    return 0;
}
