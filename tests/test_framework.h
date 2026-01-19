#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <string>
#include <vector>
#include <functional>
#include <cstdio>
#include <cmath>
#include <chrono>

// =============================================================================
// Simple Test Framework
// =============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

class TestSuite {
public:
    explicit TestSuite(const std::string& name) : name_(name), passed_(0), failed_(0) {}

    void add_test(const std::string& name, std::function<void()> test_fn) {
        tests_.push_back({name, test_fn});
    }

    void run() {
        printf("\n");
        printf("================================================================================\n");
        printf("Test Suite: %s\n", name_.c_str());
        printf("================================================================================\n");

        auto suite_start = std::chrono::steady_clock::now();

        for (auto& [name, fn] : tests_) {
            current_test_ = name;
            current_passed_ = true;
            current_message_ = "";

            auto start = std::chrono::steady_clock::now();

            try {
                fn();
            } catch (const std::exception& e) {
                current_passed_ = false;
                current_message_ = std::string("Exception: ") + e.what();
            } catch (...) {
                current_passed_ = false;
                current_message_ = "Unknown exception";
            }

            auto end = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end - start).count();

            if (current_passed_) {
                passed_++;
                printf("  [PASS] %s (%.2fms)\n", name.c_str(), duration);
            } else {
                failed_++;
                printf("  [FAIL] %s (%.2fms)\n", name.c_str(), duration);
                if (!current_message_.empty()) {
                    printf("         %s\n", current_message_.c_str());
                }
            }

            results_.push_back({name, current_passed_, current_message_, duration});
        }

        auto suite_end = std::chrono::steady_clock::now();
        double total_duration = std::chrono::duration<double, std::milli>(suite_end - suite_start).count();

        printf("--------------------------------------------------------------------------------\n");
        printf("Results: %d passed, %d failed, %zu total (%.2fms)\n",
               passed_, failed_, tests_.size(), total_duration);
        printf("================================================================================\n");
    }

    void fail(const std::string& message) {
        current_passed_ = false;
        current_message_ = message;
    }

    int passed() const { return passed_; }
    int failed() const { return failed_; }
    const std::vector<TestResult>& results() const { return results_; }

private:
    std::string name_;
    std::vector<std::pair<std::string, std::function<void()>>> tests_;
    std::vector<TestResult> results_;
    std::string current_test_;
    bool current_passed_;
    std::string current_message_;
    int passed_;
    int failed_;
};

// Global test suite pointer for macros
inline TestSuite* g_current_suite = nullptr;

// =============================================================================
// Assertion Macros
// =============================================================================

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            if (g_current_suite) g_current_suite->fail("Assertion failed: " #cond); \
            return; \
        } \
    } while(0)

#define TEST_ASSERT_MSG(cond, msg) \
    do { \
        if (!(cond)) { \
            if (g_current_suite) g_current_suite->fail(msg); \
            return; \
        } \
    } while(0)

#define TEST_ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected %s == %s", #a, #b); \
            if (g_current_suite) g_current_suite->fail(buf); \
            return; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            char buf[256]; \
            snprintf(buf, sizeof(buf), "Expected %s ~= %s (diff=%.6f, eps=%.6f)", \
                     #a, #b, std::abs((a) - (b)), (eps)); \
            if (g_current_suite) g_current_suite->fail(buf); \
            return; \
        } \
    } while(0)

#define TEST_ASSERT_SHAPE(tensor, expected_shape) \
    do { \
        if ((tensor)->shape != (expected_shape)) { \
            if (g_current_suite) g_current_suite->fail("Shape mismatch for " #tensor); \
            return; \
        } \
    } while(0)

// =============================================================================
// Test Runner
// =============================================================================

class TestRunner {
public:
    void add_suite(TestSuite* suite) {
        suites_.push_back(suite);
    }

    int run_all() {
        printf("\n");
        printf("################################################################################\n");
        printf("#                         WHITEMATTER UNIT TESTS                               #\n");
        printf("################################################################################\n");

        auto start = std::chrono::steady_clock::now();
        int total_passed = 0;
        int total_failed = 0;

        for (auto* suite : suites_) {
            g_current_suite = suite;
            suite->run();
            total_passed += suite->passed();
            total_failed += suite->failed();
        }

        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();

        printf("\n");
        printf("################################################################################\n");
        printf("TOTAL: %d passed, %d failed (%.2fs)\n", total_passed, total_failed, duration);
        printf("################################################################################\n");

        return total_failed > 0 ? 1 : 0;
    }

private:
    std::vector<TestSuite*> suites_;
};

#endif // TEST_FRAMEWORK_H
