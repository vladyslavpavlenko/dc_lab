// export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
// clang++ -std=c++20 -fopenmp main.cpp -o brutalist && ./brutalist

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdint>
#include <algorithm>

const std::string alphabet =
    "1234567890"
    "aeorisntlmdcphbukgyzfjqxvw"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!@#$%^&*()_+-=[]{}\\|;:'\",<.>/?~`";

const uint64_t k = alphabet.size();

inline void lex_inc(std::vector<uint8_t>& idxs,
                    std::string& guess,
                    const std::string& alpha) {
    for (int pos = static_cast<int>(idxs.size()) - 1; pos >= 0; --pos) {
        if (++idxs[pos] < alpha.size()) {
            guess[pos] = alpha[idxs[pos]];
            return;
        }
        idxs[pos] = 0;
        guess[pos] = alpha[0];
    }
}

int main() {
    std::string target;
    std::cout << "Enter target: ";
    std::cin  >> target;
    const int n = static_cast<int>(target.size());

    // check overflow
    uint64_t total_space = 1;
    for (int i = 0; i < n; ++i) {
        if (total_space > UINT64_MAX / k) {
            std::cerr << "Search space too large\n";
            return 1;
        }
        total_space *= k;
    }

    // map target to numeric index
    uint64_t target_idx = 0;
    for (char c : target) {
        auto pos = alphabet.find(c);
        if (pos == std::string::npos) {
            std::cerr << "Error: character '" << c << "' not in alphabet\n";
            return 1;
        }
        target_idx = target_idx * k + pos;
    }

    std::atomic found{false};
    std::atomic<uint64_t> global_count{0};
    double t0 = omp_get_wtime();

    std::thread timer([&] {
        std::cout << std::fixed << std::setprecision(1);
        while (!found.load(std::memory_order_relaxed)) {
            double e = omp_get_wtime() - t0;
            uint64_t done = global_count.load(std::memory_order_relaxed);
            double frac = done / static_cast<double>(total_space);
            double eta  = frac > 0 ? e / frac : 0.0;
            double rem  = frac > 0 ? eta - e : 0.0;
            std::cout << "\x1b[2K\r[ "
                      << e << "s | "
                      << (frac*100.0) << "% | "
                      << "ETA: " << rem << "s ]"
                      << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        double e = omp_get_wtime() - t0;
        std::cout << "\x1b[2K\r[ " << e << "s | 100% | ETA:0.0s ]\n\n";
    });

    std::vector<uint64_t> local_counts;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();

#pragma omp single
        local_counts.resize(threads);

        const uint64_t block = total_space / threads;
        const uint64_t start = static_cast<uint64_t>(tid) * block;
        const uint64_t end   = (tid == threads - 1 ? total_space : start + block);

        std::vector<uint8_t> idxs(n);
        {
            uint64_t v = start;
            for (int pos = n - 1; pos >= 0; --pos) {
                idxs[pos] = v % k;
                v /= k;
            }
        }
        std::string guess(n, ' ');
        for (int i = 0; i < n; ++i) guess[i] = alphabet[idxs[i]];

        uint64_t local = 0;
        for (uint64_t idx = start; idx < end && !found.load(std::memory_order_relaxed); ++idx) {
            ++local;
            global_count.fetch_add(1, std::memory_order_relaxed);
            if (idx == target_idx) {
                found.store(true, std::memory_order_relaxed);
                break;
            }
            lex_inc(idxs, guess, alphabet);
        }
        local_counts[tid] = local;
    }

    found.store(true, std::memory_order_relaxed);
    timer.join();

    // stats
    double elapsed = omp_get_wtime() - t0;
    uint64_t total_guesses = 0;
    for (auto c : local_counts) total_guesses += c;
    uint64_t min_g = *std::min_element(local_counts.begin(), local_counts.end());
    uint64_t max_g = *std::max_element(local_counts.begin(), local_counts.end());
    double   avg_g = static_cast<double>(total_guesses) / local_counts.size();
    uint64_t throughput = static_cast<uint64_t>(total_guesses / elapsed);

    // report
    std::cout << "Found          : " << (total_guesses ? "✓" : "✘") << '\n';
    std::cout << "Total guesses  : " << total_guesses << "\n";
    std::cout << "Guesses/sec    : " << throughput << "\n\n";
    std::cout << "Min guesses/th : " << min_g << "\n";
    std::cout << "Max guesses/th : " << max_g << "\n";
    std::cout << "Avg guesses/th : " << avg_g << "\n\n";

    for (size_t i = 0; i < local_counts.size(); ++i) {
        constexpr int BAR_WIDTH = 50;
        uint64_t c = local_counts[i];
        double frac = max_g ? double(c) / static_cast<double>(max_g) : 0.0;
        int dots = static_cast<int>(frac * BAR_WIDTH + 0.5);  // round

        std::cout << "Guesses/th" << (i+1) << "    : ";
        std::cout << c << " ";
        for (int d = 0; d < dots; ++d) std::cout << '.';
        std::cout << '\n';
    }

    return 0;
}