#include "test_framework.h"
#include "../core/tensor.h"
#include "../core/layer.h"
#include "../core/loss.h"
#include "../core/optimizer.h"

// External test suite creators (defined in individual test files)
TestSuite* create_tensor_tests();
TestSuite* create_autograd_tests();
TestSuite* create_layer_tests();
TestSuite* create_loss_tests();
TestSuite* create_optimizer_tests();

int main(int argc, char** argv) {
    // Parse command line arguments
    bool run_all = true;
    bool run_tensor = false;
    bool run_autograd = false;
    bool run_layers = false;
    bool run_loss = false;
    bool run_optimizer = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tensor" || arg == "-t") {
            run_tensor = true;
            run_all = false;
        } else if (arg == "--autograd" || arg == "-a") {
            run_autograd = true;
            run_all = false;
        } else if (arg == "--layers" || arg == "-l") {
            run_layers = true;
            run_all = false;
        } else if (arg == "--loss" || arg == "-L") {
            run_loss = true;
            run_all = false;
        } else if (arg == "--optimizer" || arg == "-o") {
            run_optimizer = true;
            run_all = false;
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: run_tests [OPTIONS]\n\n");
            printf("Options:\n");
            printf("  --tensor, -t      Run tensor operation tests\n");
            printf("  --autograd, -a    Run autograd tests\n");
            printf("  --layers, -l      Run layer tests\n");
            printf("  --loss, -L        Run loss function tests\n");
            printf("  --optimizer, -o   Run optimizer tests\n");
            printf("  --help, -h        Show this help message\n\n");
            printf("If no options are specified, all tests are run.\n");
            return 0;
        }
    }

    TestRunner runner;

    // Create and add test suites
    if (run_all || run_tensor) {
        runner.add_suite(create_tensor_tests());
    }

    if (run_all || run_autograd) {
        runner.add_suite(create_autograd_tests());
    }

    if (run_all || run_layers) {
        runner.add_suite(create_layer_tests());
    }

    if (run_all || run_loss) {
        runner.add_suite(create_loss_tests());
    }

    if (run_all || run_optimizer) {
        runner.add_suite(create_optimizer_tests());
    }

    return runner.run_all();
}
