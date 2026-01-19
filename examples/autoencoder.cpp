// Convolutional Autoencoder Example using ConvTranspose2d
// Trains on MNIST to learn compressed representations and reconstruct images

#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "mnist.h"
#include "logging.h"
#include <cstdio>
#include <cmath>

// =============================================================================
// Encoder: Compresses 28x28 image to latent vector
// =============================================================================
class Encoder : public Module {
public:
    Conv2d* conv1;
    Conv2d* conv2;
    Conv2d* conv3;
    Flatten* flatten;
    Linear* fc;

    size_t latent_dim;

    Encoder(size_t latent_dim = 32) : latent_dim(latent_dim) {
        // 28x28 -> 14x14 -> 7x7 -> 3x3 -> latent
        conv1 = new Conv2d(1, 16, 3, 2, 1);   // 28x28 -> 14x14
        conv2 = new Conv2d(16, 32, 3, 2, 1);  // 14x14 -> 7x7
        conv3 = new Conv2d(32, 64, 3, 2, 1);  // 7x7 -> 4x4 (with padding)
        flatten = new Flatten();
        fc = new Linear(64 * 4 * 4, latent_dim);
    }

    ~Encoder() {
        delete conv1;
        delete conv2;
        delete conv3;
        delete flatten;
        delete fc;
    }

    TensorPtr forward(const TensorPtr& x) override {
        auto h = conv1->forward(x)->relu();
        h = conv2->forward(h)->relu();
        h = conv3->forward(h)->relu();
        h = flatten->forward(h);
        return fc->forward(h);
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& p : conv1->parameters()) params.push_back(p);
        for (auto& p : conv2->parameters()) params.push_back(p);
        for (auto& p : conv3->parameters()) params.push_back(p);
        for (auto& p : fc->parameters()) params.push_back(p);
        return params;
    }

    std::string name() const override { return "Encoder"; }
};

// =============================================================================
// Decoder: Reconstructs image from latent vector using ConvTranspose2d
// =============================================================================
class Decoder : public Module {
public:
    Linear* fc;
    ConvTranspose2d* deconv1;
    ConvTranspose2d* deconv2;
    ConvTranspose2d* deconv3;

    size_t latent_dim;

    Decoder(size_t latent_dim = 32) : latent_dim(latent_dim) {
        // latent -> 4x4 -> 7x7 -> 14x14 -> 28x28
        // ConvTranspose2d output: (input - 1) * stride - 2 * padding + kernel_size + output_padding
        fc = new Linear(latent_dim, 64 * 4 * 4);
        deconv1 = new ConvTranspose2d(64, 32, 3, 2, 1, 0);  // 4x4 -> 7x7: (4-1)*2 - 2 + 3 = 7
        deconv2 = new ConvTranspose2d(32, 16, 4, 2, 1, 0);  // 7x7 -> 14x14: (7-1)*2 - 2 + 4 = 14
        deconv3 = new ConvTranspose2d(16, 1, 4, 2, 1, 0);   // 14x14 -> 28x28: (14-1)*2 - 2 + 4 = 28
    }

    ~Decoder() {
        delete fc;
        delete deconv1;
        delete deconv2;
        delete deconv3;
    }

    TensorPtr forward(const TensorPtr& z) override {
        auto h = fc->forward(z)->relu();

        // Reshape to [batch, 64, 4, 4]
        size_t batch = z->shape[0];
        h = h->reshape({batch, 64, 4, 4});

        h = deconv1->forward(h)->relu();
        h = deconv2->forward(h)->relu();
        h = deconv3->forward(h)->sigmoid();  // Output in [0, 1]

        return h;
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& p : fc->parameters()) params.push_back(p);
        for (auto& p : deconv1->parameters()) params.push_back(p);
        for (auto& p : deconv2->parameters()) params.push_back(p);
        for (auto& p : deconv3->parameters()) params.push_back(p);
        return params;
    }

    std::string name() const override { return "Decoder"; }
};

// =============================================================================
// Autoencoder: Combines Encoder and Decoder
// =============================================================================
class Autoencoder : public Module {
public:
    Encoder* encoder;
    Decoder* decoder;
    size_t latent_dim;

    Autoencoder(size_t latent_dim = 32) : latent_dim(latent_dim) {
        encoder = new Encoder(latent_dim);
        decoder = new Decoder(latent_dim);
    }

    ~Autoencoder() {
        delete encoder;
        delete decoder;
    }

    TensorPtr forward(const TensorPtr& x) override {
        auto z = encoder->forward(x);
        return decoder->forward(z);
    }

    // Get latent representation
    TensorPtr encode(const TensorPtr& x) {
        return encoder->forward(x);
    }

    // Reconstruct from latent
    TensorPtr decode(const TensorPtr& z) {
        return decoder->forward(z);
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> params;
        for (auto& p : encoder->parameters()) params.push_back(p);
        for (auto& p : decoder->parameters()) params.push_back(p);
        return params;
    }

    std::string name() const override { return "Autoencoder"; }
};

// =============================================================================
// Helper: Print a digit as ASCII art
// =============================================================================
void print_digit(const TensorPtr& img, [[maybe_unused]] int row_offset = 0) {
    // img shape: [1, 1, 28, 28] or [1, 28, 28]
    const char* chars = " .:-=+*#%@";
    int num_chars = 10;

    size_t h = 28, w = 28;
    size_t offset = (img->shape.size() == 4) ? img->shape[0] * img->shape[1] * h * w : 0;
    if (offset > 0) offset = 0;  // Start from beginning for single image

    for (size_t i = 0; i < h; i += 2) {  // Skip every other row for aspect ratio
        printf("    ");
        for (size_t j = 0; j < w; j++) {
            float val = img->data[i * w + j];
            int idx = static_cast<int>(val * (num_chars - 1));
            idx = std::max(0, std::min(num_chars - 1, idx));
            printf("%c", chars[idx]);
        }
        printf("\n");
    }
}

// =============================================================================
// Helper: Calculate reconstruction error
// =============================================================================
float reconstruction_error(Autoencoder& model, const TensorPtr& images) {
    NoGradGuard no_grad;
    auto recon = model.forward(images);

    float mse = 0.0f;
    for (size_t i = 0; i < images->data.size(); i++) {
        float diff = images->data[i] - recon->data[i];
        mse += diff * diff;
    }
    return mse / images->data.size();
}

// =============================================================================
// Main Training Loop
// =============================================================================
int main() {
    printf("=======================================================\n");
    printf("Convolutional Autoencoder - MNIST\n");
    printf("=======================================================\n\n");

    // Hyperparameters
    const size_t latent_dim = 32;
    const size_t batch_size = 64;
    const size_t num_epochs = 10;
    const float learning_rate = 0.001f;

    printf("Configuration:\n");
    printf("  Latent dimension: %zu\n", latent_dim);
    printf("  Batch size: %zu\n", batch_size);
    printf("  Epochs: %zu\n", num_epochs);
    printf("  Learning rate: %.4f\n\n", learning_rate);

    // Load MNIST data
    printf("Loading MNIST dataset...\n");
    auto train_data = load_mnist_train("data/");
    auto test_data = load_mnist_test("data/");
    printf("  Training samples: %zu\n", train_data.num_samples);
    printf("  Test samples: %zu\n\n", test_data.num_samples);

    // Create model
    Autoencoder model(latent_dim);

    // Count parameters
    size_t total_params = 0;
    for (auto& p : model.parameters()) {
        total_params += p->data.size();
    }
    printf("Model Parameters: %zu\n\n", total_params);

    // Optimizer and loss
    Adam optimizer(model.parameters(), learning_rate);
    MSELoss criterion;

    // Training logger
    TrainingLogger logger("logs", "autoencoder");
    logger.set_total_epochs(num_epochs);

    // Create data loaders
    DataLoader train_loader(train_data, batch_size, true);
    DataLoader test_loader(test_data, batch_size, false);

    // Training loop
    printf("Training...\n");
    printf("-------------------------------------------------------\n");

    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        logger.new_epoch();
        train_loader.reset();

        float epoch_loss = 0.0f;
        size_t num_batches = 0;

        while (train_loader.has_next()) {
            auto [images, labels] = train_loader.next_batch();

            // Reshape to [batch, 1, 28, 28] for conv layers
            size_t current_batch = images->shape[0];
            images = images->reshape({current_batch, 1, 28, 28});

            // Forward pass
            optimizer.zero_grad();
            auto recon = model.forward(images);
            auto loss = criterion(recon, images);

            // Backward pass
            loss->backward();
            optimizer.step();

            epoch_loss += loss->item();
            num_batches++;

            logger.log_batch("loss", loss->item());
        }

        epoch_loss /= num_batches;

        // Evaluate on test set
        test_loader.reset();
        float test_loss = 0.0f;
        size_t test_batches = 0;

        {
            NoGradGuard no_grad;
            while (test_loader.has_next()) {
                auto [images, labels] = test_loader.next_batch();
                size_t current_batch = images->shape[0];
                images = images->reshape({current_batch, 1, 28, 28});

                auto recon = model.forward(images);
                auto loss = criterion(recon, images);
                test_loss += loss->item();
                test_batches++;
            }
        }
        test_loss /= test_batches;

        logger.log("train_loss", epoch_loss);
        logger.log("test_loss", test_loss);
        logger.step();

        printf("Epoch %2zu/%zu  Train Loss: %.6f  Test Loss: %.6f\n",
               epoch + 1, num_epochs, epoch_loss, test_loss);
    }

    printf("-------------------------------------------------------\n\n");

    // Show reconstruction examples
    printf("Reconstruction Examples:\n");
    printf("=======================================================\n");

    test_loader.reset();
    auto [test_images, test_labels] = test_loader.next_batch();

    // Show a few examples
    for (int i = 0; i < 3; i++) {
        // Extract single image
        auto single_img = Tensor::create({1, 1, 28, 28}, false);
        for (size_t j = 0; j < 784; j++) {
            single_img->data[j] = test_images->data[i * 784 + j];
        }

        // Reconstruct
        TensorPtr recon;
        {
            NoGradGuard no_grad;
            recon = model.forward(single_img);
        }

        // Flatten for printing
        auto orig_flat = single_img->reshape({28, 28});
        auto recon_flat = recon->reshape({28, 28});

        printf("\nExample %d (Label: %.0f)\n", i + 1, test_labels->data[i]);
        printf("Original:                    Reconstructed:\n");

        const char* chars = " .:-=+*#%@";
        int num_chars = 10;

        for (size_t row = 0; row < 28; row += 2) {
            printf("    ");
            // Original
            for (size_t col = 0; col < 28; col++) {
                float val = orig_flat->data[row * 28 + col];
                int idx = static_cast<int>(val * (num_chars - 1));
                idx = std::max(0, std::min(num_chars - 1, idx));
                printf("%c", chars[idx]);
            }
            printf("    ");
            // Reconstructed
            for (size_t col = 0; col < 28; col++) {
                float val = recon_flat->data[row * 28 + col];
                int idx = static_cast<int>(val * (num_chars - 1));
                idx = std::max(0, std::min(num_chars - 1, idx));
                printf("%c", chars[idx]);
            }
            printf("\n");
        }
    }

    printf("\n=======================================================\n");

    // Show latent space interpolation
    printf("\nLatent Space Interpolation:\n");
    printf("=======================================================\n");

    // Get two different digits and interpolate
    auto img1 = Tensor::create({1, 1, 28, 28}, false);
    auto img2 = Tensor::create({1, 1, 28, 28}, false);

    // Find two different digits
    int idx1 = 0, idx2 = 1;
    for (size_t i = 0; i < test_images->shape[0] && idx2 <= idx1; i++) {
        if (test_labels->data[i] != test_labels->data[idx1]) {
            idx2 = i;
        }
    }

    for (size_t j = 0; j < 784; j++) {
        img1->data[j] = test_images->data[idx1 * 784 + j];
        img2->data[j] = test_images->data[idx2 * 784 + j];
    }

    printf("Interpolating between digit %.0f and digit %.0f:\n\n",
           test_labels->data[idx1], test_labels->data[idx2]);

    TensorPtr z1, z2;
    {
        NoGradGuard no_grad;
        z1 = model.encode(img1);
        z2 = model.encode(img2);
    }

    // Generate interpolated images
    std::vector<TensorPtr> interpolations;
    int num_steps = 5;

    for (int step = 0; step < num_steps; step++) {
        float alpha = static_cast<float>(step) / (num_steps - 1);

        // Linear interpolation in latent space
        auto z_interp = Tensor::create({1, latent_dim}, false);
        for (size_t i = 0; i < latent_dim; i++) {
            z_interp->data[i] = (1.0f - alpha) * z1->data[i] + alpha * z2->data[i];
        }

        TensorPtr decoded;
        {
            NoGradGuard no_grad;
            decoded = model.decode(z_interp);
        }
        interpolations.push_back(decoded->reshape({28, 28}));
    }

    // Print interpolations side by side
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
            printf("  ");
        }
        printf("\n");
    }

    printf("\n=======================================================\n");
    printf("Training complete!\n");
    printf("=======================================================\n");

    // Save logs
    logger.save_csv();
    logger.save_json();

    return 0;
}
