// mpic++ -std=c++20 -O3 -o brutto main.cpp
// mpirun -np 8 ./brutto

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>

const std::string alphabet =
    "1234567890"
    "aeorisntlmdcphbukgyzfjqxvw"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!@#$%^&*()_+-=[]{}\\|;:'\",<.>/?~`";
const uint64_t k = alphabet.size();

inline void lex_inc(std::vector<uint8_t>& idxs,
                    std::string& guess,
                    const std::string& alpha) {
    int n = static_cast<int>(idxs.size());
    for (int pos = n - 1; pos >= 0; --pos) {
        if (++idxs[pos] < alpha.size()) {
            guess[pos] = alpha[idxs[pos]];
            return;
        }
        idxs[pos] = 0;
        guess[pos] = alpha[0];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string target;
    int n;
    if (rank == 0) {
        std::cout << "Enter target: " << std::endl;
        std::cin >> target;
        n = static_cast<int>(target.size());
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) target.resize(n);
    MPI_Bcast(const_cast<char*>(target.data()), n, MPI_CHAR, 0, MPI_COMM_WORLD);

    uint64_t total_space = 1;
    for (int i = 0; i < n; ++i) {
        if (total_space > UINT64_MAX / k) {
            if (rank == 0) std::cerr << "Search space too large" << std::endl;
            MPI_Finalize();
            return 1;
        }
        total_space *= k;
    }

    uint64_t target_idx = 0;
    for (char c : target) {
        size_t pos = alphabet.find(c);
        if (pos == std::string::npos) {
            if (rank == 0) std::cerr << "Error: character '" << c << "' not in alphabet" << std::endl;
            MPI_Finalize();
            return 1;
        }
        target_idx = target_idx * k + pos;
    }

    uint64_t local_count = 0;
    bool local_found = false;

    double t0 = MPI_Wtime();
    std::vector<uint8_t> idxs(n);
    std::string guess(n, ' ');
    int any_found = 0;

    for (uint64_t idx = rank; idx < total_space; idx += size) {
        // decode idx -> guess
        uint64_t v = idx;
        for (int pos = n - 1; pos >= 0; --pos) {
            idxs[pos] = v % k;
            v /= k;
        }
        for (int i = 0; i < n; ++i) guess[i] = alphabet[idxs[i]];

        ++local_count;
        if (idx == target_idx) {
            local_found = true;
        }
        int lf_int = local_found ? 1 : 0;
        MPI_Allreduce(&lf_int, &any_found, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (any_found) break;
    }
    double t1 = MPI_Wtime();

    std::vector<uint64_t> counts(size);
    MPI_Gather(&local_count, 1, MPI_UNSIGNED_LONG_LONG,
               counts.data(),    1, MPI_UNSIGNED_LONG_LONG,
               0, MPI_COMM_WORLD);

    if (rank==0) {
        uint64_t total_guesses = 0;
        for (auto c : counts) total_guesses += c;
    }
    int lf_int = local_found ? 1 : 0;
    int any_found_reduce = 0;
    MPI_Reduce(&lf_int, &any_found_reduce, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = t1 - t0;
        uint64_t total_guesses = 0;
        for (auto c : counts) total_guesses += c;
        uint64_t throughput = static_cast<uint64_t>(total_guesses / elapsed);

        std::cout << "Found         : " << (any_found_reduce ? "✓" : "✘") << std::endl;
        std::cout << "Elapsed time  : " << elapsed << 's' << std::endl;
        std::cout << "Total guesses : " << total_guesses << std::endl;
        std::cout << "Guesses/sec   : " << throughput << std::endl;
    }

    MPI_Finalize();
    return 0;
}
