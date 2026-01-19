#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

// =============================================================================
// MetricTracker - Track running statistics for a single metric
// =============================================================================

class MetricTracker {
public:
    MetricTracker() { reset(); }

    void update(float value) {
        values_.push_back(value);
        sum_ += value;
        count_++;
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);

        // Update running mean and variance (Welford's algorithm)
        float delta = value - mean_;
        mean_ += delta / count_;
        float delta2 = value - mean_;
        m2_ += delta * delta2;
    }

    void reset() {
        values_.clear();
        sum_ = 0.0f;
        count_ = 0;
        min_ = std::numeric_limits<float>::infinity();
        max_ = -std::numeric_limits<float>::infinity();
        mean_ = 0.0f;
        m2_ = 0.0f;
    }

    float sum() const { return sum_; }
    float mean() const { return count_ > 0 ? sum_ / count_ : 0.0f; }
    float min() const { return count_ > 0 ? min_ : 0.0f; }
    float max() const { return count_ > 0 ? max_ : 0.0f; }
    float last() const { return values_.empty() ? 0.0f : values_.back(); }
    size_t count() const { return count_; }

    float variance() const {
        return count_ > 1 ? m2_ / (count_ - 1) : 0.0f;
    }

    float std() const {
        return std::sqrt(variance());
    }

    const std::vector<float>& values() const { return values_; }

private:
    std::vector<float> values_;
    float sum_;
    size_t count_;
    float min_;
    float max_;
    float mean_;
    float m2_;  // For Welford's variance algorithm
};

// =============================================================================
// LogEntry - Single log entry with step, timestamp, and metrics
// =============================================================================

struct LogEntry {
    int step;
    int epoch;
    double timestamp;  // Seconds since training start
    std::map<std::string, float> metrics;
};

// =============================================================================
// TrainingLogger - TensorBoard-style logging for training metrics
// =============================================================================

class TrainingLogger {
public:
    // Create logger with optional log directory
    // If log_dir is empty, only console output is used
    explicit TrainingLogger(const std::string& log_dir = "",
                           const std::string& experiment_name = "run")
        : log_dir_(log_dir),
          experiment_name_(experiment_name),
          current_step_(0),
          current_epoch_(0),
          total_steps_(0),
          total_epochs_(0),
          verbose_(true),
          console_width_(80) {
        start_time_ = std::chrono::steady_clock::now();

        if (!log_dir_.empty()) {
            // Create log directory if needed (simplified - assumes directory exists)
            csv_path_ = log_dir_ + "/" + experiment_name_ + "_metrics.csv";
            json_path_ = log_dir_ + "/" + experiment_name_ + "_metrics.json";
        }
    }

    // Set total steps/epochs for progress display
    void set_total_steps(int total) { total_steps_ = total; }
    void set_total_epochs(int total) { total_epochs_ = total; }

    // Set verbosity
    void set_verbose(bool v) { verbose_ = v; }

    // Log a scalar metric at current step
    void log(const std::string& name, float value) {
        // Update tracker
        if (trackers_.find(name) == trackers_.end()) {
            trackers_[name] = MetricTracker();
            metric_names_.push_back(name);
        }
        trackers_[name].update(value);

        // Add to pending metrics for current step
        pending_metrics_[name] = value;
    }

    // Log multiple metrics at once
    void log(const std::map<std::string, float>& metrics) {
        for (const auto& [name, value] : metrics) {
            log(name, value);
        }
    }

    // Commit pending metrics as a log entry and advance step
    void step() {
        if (!pending_metrics_.empty()) {
            LogEntry entry;
            entry.step = current_step_;
            entry.epoch = current_epoch_;
            entry.timestamp = elapsed_seconds();
            entry.metrics = pending_metrics_;
            history_.push_back(entry);
            pending_metrics_.clear();
        }
        current_step_++;
    }

    // Start a new epoch
    void new_epoch() {
        // Reset epoch trackers
        for (auto& [name, tracker] : epoch_trackers_) {
            tracker.reset();
        }
        current_epoch_++;
    }

    // Log metric for epoch-level aggregation
    void log_batch(const std::string& name, float value) {
        if (epoch_trackers_.find(name) == epoch_trackers_.end()) {
            epoch_trackers_[name] = MetricTracker();
        }
        epoch_trackers_[name].update(value);
    }

    // Get epoch average for a metric
    float epoch_mean(const std::string& name) const {
        auto it = epoch_trackers_.find(name);
        return it != epoch_trackers_.end() ? it->second.mean() : 0.0f;
    }

    // Print progress bar with current metrics
    void print_progress(const std::string& prefix = "") {
        if (!verbose_) return;

        std::string line = prefix;

        // Add epoch/step info
        if (total_epochs_ > 0) {
            line += "Epoch " + std::to_string(current_epoch_) + "/" +
                    std::to_string(total_epochs_) + " ";
        }

        // Add progress bar if total_steps known
        if (total_steps_ > 0) {
            int progress = (current_step_ % total_steps_);
            if (progress == 0 && current_step_ > 0) progress = total_steps_;
            float pct = static_cast<float>(progress) / total_steps_;
            int bar_width = 20;
            int filled = static_cast<int>(pct * bar_width);
            line += "[";
            for (int i = 0; i < bar_width; i++) {
                line += (i < filled) ? "=" : (i == filled ? ">" : " ");
            }
            line += "] ";
            line += std::to_string(static_cast<int>(pct * 100)) + "% ";
        }

        // Add metrics
        for (const auto& [name, tracker] : epoch_trackers_) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: %.4f ", name.c_str(), tracker.mean());
            line += buf;
        }

        // Add elapsed time
        line += "(" + format_time(elapsed_seconds()) + ")";

        // Print with carriage return for in-place update
        printf("\r%-*s", console_width_, line.c_str());
        fflush(stdout);
    }

    // Print epoch summary
    void print_epoch_summary() {
        if (!verbose_) return;
        printf("\n");

        std::string line = "Epoch " + std::to_string(current_epoch_);
        if (total_epochs_ > 0) {
            line += "/" + std::to_string(total_epochs_);
        }
        line += " - ";

        for (const auto& [name, tracker] : epoch_trackers_) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: %.4f ", name.c_str(), tracker.mean());
            line += buf;
        }

        line += "- " + format_time(elapsed_seconds());
        printf("%s\n", line.c_str());
    }

    // Get metric tracker
    const MetricTracker& get_tracker(const std::string& name) const {
        static MetricTracker empty;
        auto it = trackers_.find(name);
        return it != trackers_.end() ? it->second : empty;
    }

    // Get all history
    const std::vector<LogEntry>& history() const { return history_; }

    // Get current step/epoch
    int current_step() const { return current_step_; }
    int current_epoch() const { return current_epoch_; }

    // Get elapsed time in seconds
    double elapsed_seconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    // Save logs to CSV file
    bool save_csv(const std::string& path = "") const {
        std::string filepath = path.empty() ? csv_path_ : path;
        if (filepath.empty()) return false;

        std::ofstream out(filepath);
        if (!out) return false;

        // Write header
        out << "step,epoch,timestamp";
        for (const auto& name : metric_names_) {
            out << "," << name;
        }
        out << "\n";

        // Write data
        for (const auto& entry : history_) {
            out << entry.step << "," << entry.epoch << ","
                << std::fixed << std::setprecision(4) << entry.timestamp;
            for (const auto& name : metric_names_) {
                out << ",";
                auto it = entry.metrics.find(name);
                if (it != entry.metrics.end()) {
                    out << std::fixed << std::setprecision(6) << it->second;
                }
            }
            out << "\n";
        }

        return true;
    }

    // Save logs to JSON file
    bool save_json(const std::string& path = "") const {
        std::string filepath = path.empty() ? json_path_ : path;
        if (filepath.empty()) return false;

        std::ofstream out(filepath);
        if (!out) return false;

        out << "{\n";
        out << "  \"experiment\": \"" << experiment_name_ << "\",\n";
        out << "  \"total_steps\": " << current_step_ << ",\n";
        out << "  \"total_epochs\": " << current_epoch_ << ",\n";
        out << "  \"elapsed_seconds\": " << std::fixed << std::setprecision(2)
            << elapsed_seconds() << ",\n";

        // Write summary statistics
        out << "  \"summary\": {\n";
        bool first_metric = true;
        for (const auto& name : metric_names_) {
            if (!first_metric) out << ",\n";
            first_metric = false;
            const auto& tracker = trackers_.at(name);
            out << "    \"" << name << "\": {\n";
            out << "      \"min\": " << tracker.min() << ",\n";
            out << "      \"max\": " << tracker.max() << ",\n";
            out << "      \"mean\": " << tracker.mean() << ",\n";
            out << "      \"std\": " << tracker.std() << ",\n";
            out << "      \"last\": " << tracker.last() << "\n";
            out << "    }";
        }
        out << "\n  },\n";

        // Write history
        out << "  \"history\": [\n";
        for (size_t i = 0; i < history_.size(); i++) {
            const auto& entry = history_[i];
            out << "    {\"step\": " << entry.step
                << ", \"epoch\": " << entry.epoch
                << ", \"timestamp\": " << std::fixed << std::setprecision(4)
                << entry.timestamp;
            for (const auto& [name, value] : entry.metrics) {
                out << ", \"" << name << "\": " << std::fixed << std::setprecision(6) << value;
            }
            out << "}" << (i < history_.size() - 1 ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        return true;
    }

    // Print final summary
    void print_summary() const {
        printf("\n");
        printf("==============================================================================\n");
        printf("Training Summary: %s\n", experiment_name_.c_str());
        printf("==============================================================================\n");
        printf("Total epochs:  %d\n", current_epoch_);
        printf("Total steps:   %d\n", current_step_);
        printf("Elapsed time:  %s\n", format_time(elapsed_seconds()).c_str());
        printf("------------------------------------------------------------------------------\n");

        for (const auto& name : metric_names_) {
            const auto& tracker = trackers_.at(name);
            printf("%-15s min: %8.4f  max: %8.4f  mean: %8.4f  std: %8.4f\n",
                   (name + ":").c_str(), tracker.min(), tracker.max(),
                   tracker.mean(), tracker.std());
        }
        printf("==============================================================================\n");
    }

    // Reset logger
    void reset() {
        trackers_.clear();
        epoch_trackers_.clear();
        metric_names_.clear();
        history_.clear();
        pending_metrics_.clear();
        current_step_ = 0;
        current_epoch_ = 0;
        start_time_ = std::chrono::steady_clock::now();
    }

private:
    std::string format_time(double seconds) const {
        int hrs = static_cast<int>(seconds / 3600);
        int mins = static_cast<int>((seconds - hrs * 3600) / 60);
        int secs = static_cast<int>(seconds) % 60;

        char buf[32];
        if (hrs > 0) {
            snprintf(buf, sizeof(buf), "%dh %dm %ds", hrs, mins, secs);
        } else if (mins > 0) {
            snprintf(buf, sizeof(buf), "%dm %ds", mins, secs);
        } else {
            snprintf(buf, sizeof(buf), "%.1fs", seconds);
        }
        return std::string(buf);
    }

    std::string log_dir_;
    std::string experiment_name_;
    std::string csv_path_;
    std::string json_path_;

    int current_step_;
    int current_epoch_;
    int total_steps_;
    int total_epochs_;
    bool verbose_;
    int console_width_;

    std::chrono::steady_clock::time_point start_time_;

    std::map<std::string, MetricTracker> trackers_;
    std::map<std::string, MetricTracker> epoch_trackers_;
    std::vector<std::string> metric_names_;
    std::vector<LogEntry> history_;
    std::map<std::string, float> pending_metrics_;
};

// =============================================================================
// ProgressBar - Simple progress bar for loops
// =============================================================================

class ProgressBar {
public:
    ProgressBar(int total, int width = 40, const std::string& prefix = "")
        : total_(total), current_(0), width_(width), prefix_(prefix) {
        start_time_ = std::chrono::steady_clock::now();
    }

    void update(int n = 1) {
        current_ += n;
        display();
    }

    void set(int n) {
        current_ = n;
        display();
    }

    void finish() {
        current_ = total_;
        display();
        printf("\n");
    }

private:
    void display() {
        float pct = static_cast<float>(current_) / total_;
        int filled = static_cast<int>(pct * width_);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();
        double eta = (current_ > 0) ? elapsed * (total_ - current_) / current_ : 0;

        printf("\r%s[", prefix_.c_str());
        for (int i = 0; i < width_; i++) {
            if (i < filled) printf("=");
            else if (i == filled) printf(">");
            else printf(" ");
        }
        printf("] %3d%% %d/%d [%.1fs < %.1fs]",
               static_cast<int>(pct * 100), current_, total_, elapsed, eta);
        fflush(stdout);
    }

    int total_;
    int current_;
    int width_;
    std::string prefix_;
    std::chrono::steady_clock::time_point start_time_;
};

// =============================================================================
// Convenience function for quick logging setup
// =============================================================================

inline TrainingLogger create_logger(const std::string& experiment_name,
                                    const std::string& log_dir = "logs") {
    return TrainingLogger(log_dir, experiment_name);
}

#endif // LOGGING_H
