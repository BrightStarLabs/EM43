// $ g++ em43_all.cpp -Ofast -march=znver2 -flto -funroll-loops -fomit-frame-pointer -fopenmp -std=c++17 -o em43_ga
// $ ./em43_ga
// -> depending on your CPU, optimizations may require different parameters
// Author: Giacomo Bocchese - with the help of ChatGPT
// This code has not been checked - may still present unexpected behaviours

#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace em43 {

// Basic types
constexpr std::size_t LUT_SIZE = 64;
using Rule      = std::array<std::uint8_t, LUT_SIZE>;
using Programme = std::vector<std::uint8_t>;

constexpr std::uint8_t S_BLANK = 0;
constexpr std::uint8_t S_PROG  = 1;
constexpr std::uint8_t S_R     = 2;
constexpr std::uint8_t S_B     = 3;

inline constexpr std::array<std::pair<std::uint8_t,std::uint8_t>,8> IMMUTABLE {{
    { (0<<4)|(0<<2)|0     , S_BLANK },
    { (0<<4)|(S_R<<2)|0   , S_R     },
    { (0<<4)|(0<<2)|S_R   , S_BLANK },
    { (S_R<<4)|(0<<2)|0   , S_BLANK },
    { (0<<4)|(S_B<<2)|S_B , S_B     },
    { (S_B<<4)|(S_B<<2)|0 , S_B     },
    { (0<<4)|(0<<2)|S_B   , S_BLANK },
    { (S_B<<4)|(0<<2)|0   , S_BLANK }
}};

inline void sanitize_rule(Rule& r) {
    for (auto& e : r) e &= 0x03u;
    for (auto [idx, val] : IMMUTABLE) r[idx] = val;
}
inline void sanitize_programme(Programme& p) {
    std::replace(p.begin(), p.end(), S_B, S_BLANK);
}

struct SimParams {
    std::size_t L, N;
    int max_steps;
    double halt_thresh;
};

int simulate_single(const Rule& rule, const Programme& prog, int n_in, const SimParams& P) {
    std::size_t L = P.L, N = P.N;
    std::vector<std::uint8_t> cur(N, 0), nxt(N, 0), frozen(N, 0);

    std::copy(prog.begin(), prog.end(), cur.begin());
    cur[L] = cur[L+1] = S_B;
    std::size_t r_idx = L + 2 + static_cast<std::size_t>(n_in) + 1;
    if (r_idx >= N) return -10;
    cur[r_idx] = S_R;

    for (int step = 0; step < P.max_steps; ++step) {
        bool any_live = false; int live = 0, blue = 0;

        for (std::size_t x = 1; x+1 < N; ++x) {
            std::uint8_t idx = (cur[x-1] << 4) | (cur[x] << 2) | cur[x+1];
            nxt[x] = rule[idx];
        }
        nxt[0] = nxt[N-1] = S_BLANK;

        for (auto v : nxt) {
            if (v != S_BLANK) { any_live = true; ++live; if (v == S_B) ++blue; }
        }
        if (any_live && static_cast<double>(blue) / live >= P.halt_thresh) {
            frozen.swap(nxt);
            break;
        }
        cur.swap(nxt);
    }

    for (std::size_t x = N; x-- > 0; )
        if (frozen[x] == S_R)
            return static_cast<int>(x) - static_cast<int>(L + 3);
    return -10;
}

class Batch {
public:
    Batch(Rule r, Programme p, std::size_t win=500, int steps=256, double halt=0.5)
        : rule_(std::move(r)), prog_(std::move(p)), params_{prog_.size(), win, steps, halt} {
        sanitize_rule(rule_);
        sanitize_programme(prog_);
        if (prog_.size() + 5 >= win)
            throw std::invalid_argument("window too small for programme");
    }

    std::vector<int> run(const std::vector<int>& inputs) const {
        std::vector<int> out(inputs.size(), -10);
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(inputs.size()); ++i)
            out[i] = simulate_single(rule_, prog_, inputs[i], params_);
        return out;
    }

private:
    Rule rule_; Programme prog_; SimParams params_;
};

std::vector<float> evaluate_population(
    const std::vector<Rule>& rules,
    const std::vector<Programme>& progs,
    const std::vector<int>& inputs,
    const std::vector<int>& targets,
    double lambda_p, std::size_t win, int steps, double halt)
{
    std::vector<float> fit(rules.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(rules.size()); ++i) {
        Batch sim(rules[i], progs[i], win, steps, halt);
        auto out = sim.run(inputs);
        double err = 0.0;
        for (std::size_t k = 0; k < out.size(); ++k) err += std::abs(out[k] - targets[k]);
        double sparsity = std::count_if(progs[i].begin(), progs[i].end(), [](uint8_t v){return v != 0;}) / static_cast<double>(progs[i].size());
        fit[i] = static_cast<float>(-err / out.size() - lambda_p * sparsity);
    }
    return fit;
}

} // namespace em43

// === GA Implementation ===

using em43::Rule;
using em43::Programme;
namespace fs = std::filesystem;

constexpr std::size_t POP_SIZE = 20000;
constexpr int N_GENERATIONS = 300;
constexpr double ELITE_FRAC = 0.10;
constexpr int TOURNEY_K = 3;
constexpr double P_MUT_RULE = 0.02;
constexpr double P_MUT_PROG = 0.06;
constexpr std::size_t L_PROG = 32;
constexpr double LAMBDA_P = 0.01;
constexpr double EPS_IMMIGR = 0.05;
constexpr std::size_t WINDOW = 200;
constexpr int MAX_STEPS = 500;
constexpr double HALT_THRESH = 0.50;
constexpr int CHECK_EVERY = 50;

std::mt19937_64 rng(std::random_device{}());
std::uniform_real_distribution<> unif01(0.0, 1.0);
std::uniform_int_distribution<> cell_dist(0, 3);
std::discrete_distribution<> prog_dist({0.7, 0.2, 0.1});

Rule random_rule() {
    Rule r; for (auto& e : r) e = cell_dist(rng); em43::sanitize_rule(r); return r;
}
Programme random_prog() {
    Programme p(L_PROG); for (auto& e : p) e = prog_dist(rng); em43::sanitize_programme(p); return p;
}
void mutate(Rule& r, Programme& p) {
    for (auto& e : r) if (unif01(rng) < P_MUT_RULE) e = cell_dist(rng);
    em43::sanitize_rule(r);
    for (auto& e : p) if (unif01(rng) < P_MUT_PROG) e = prog_dist(rng);
    em43::sanitize_programme(p);
}
std::pair<Rule, Programme> crossover(const Rule& r1, const Programme& p1, const Rule& r2, const Programme& p2) {
    std::array<uint8_t, 64 + L_PROG> a1, a2;
    std::copy(r1.begin(), r1.end(), a1.begin());
    std::copy(p1.begin(), p1.end(), a1.begin() + 64);
    std::copy(r2.begin(), r2.end(), a2.begin());
    std::copy(p2.begin(), p2.end(), a2.begin() + 64);

    std::uniform_int_distribution<> cut_dist(1, 64 + L_PROG - 1);
    int cut = cut_dist(rng);
    for (int i = cut; i < static_cast<int>(64 + L_PROG); ++i) std::swap(a1[i], a2[i]);

    Rule child_r;
    Programme child_p(L_PROG);
    std::copy(a1.begin(), a1.begin() + 64, child_r.begin());
    std::copy(a1.begin() + 64, a1.end(), child_p.begin());
    em43::sanitize_rule(child_r);
    em43::sanitize_programme(child_p);
    return {child_r, child_p};
}
std::size_t tournament(const std::vector<float>& fit) {
    std::uniform_int_distribution<> idx(0, POP_SIZE - 1);
    std::size_t best = idx(rng);
    for (int i = 1; i < TOURNEY_K; ++i) {
        std::size_t j = idx(rng);
        if (fit[j] > fit[best]) best = j;
    }
    return best;
}

const std::vector<int> INPUT_SET = []{ std::vector<int> v(30); std::iota(v.begin(), v.end(), 1); return v; }();
const std::vector<int> TARGET_OUT = []{ std::vector<int> v(30); for (int i = 0; i < 30; ++i) v[i] = 4*(i+1); return v; }();

int main() {
    #ifdef _OPENMP
        std::cout<<"_OPENMP="<<_OPENMP<<"\n";
        std::cout<<"omp threads="<<omp_get_max_threads()<<"\n";
    #else
        std::cout<<"OpenMP not enabled!\n";
    #endif

    fs::create_directory("dp_checkpoints");
    std::vector<Rule> rules(POP_SIZE); std::vector<Programme> progs(POP_SIZE);
    for (std::size_t i = 0; i < POP_SIZE; ++i) {
        rules[i] = random_rule(); progs[i] = random_prog();
    }

    const std::size_t N_ELITE = std::ceil(ELITE_FRAC * POP_SIZE);
    const std::size_t N_IMM = std::ceil(EPS_IMMIGR * POP_SIZE);
    std::vector<float> best_curve, mean_curve;

    for (int gen = 1; gen <= N_GENERATIONS; ++gen) {
        auto fit = em43::evaluate_population(rules, progs, INPUT_SET, TARGET_OUT, LAMBDA_P, WINDOW, MAX_STEPS, HALT_THRESH);
        std::vector<std::size_t> order(POP_SIZE); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b){ return fit[a] > fit[b]; });

        std::vector<Rule> new_rules(POP_SIZE); std::vector<Programme> new_progs(POP_SIZE);
        for (std::size_t i = 0; i < POP_SIZE; ++i) {
            new_rules[i] = rules[order[i]];
            new_progs[i] = progs[order[i]];
        }
        rules.swap(new_rules); progs.swap(new_progs);

        float best = fit[order[0]];
        float mean = std::accumulate(fit.begin(), fit.end(), 0.0f) / POP_SIZE;
        best_curve.push_back(best); mean_curve.push_back(mean);
        std::cout << "Gen " << gen << "  best=" << best << "  mean=" << mean << "\n";

        if (gen % CHECK_EVERY == 0 || gen == N_GENERATIONS) {
            std::ofstream ck("dp_checkpoints/ckpt_gen" + std::to_string(gen) + ".bin", std::ios::binary);
            ck.write(reinterpret_cast<const char*>(rules[0].data()), rules[0].size());
            ck.write(reinterpret_cast<const char*>(progs[0].data()), progs[0].size());
        }

        std::vector<Rule> next_rules(rules.begin(), rules.begin() + N_ELITE);
        std::vector<Programme> next_progs(progs.begin(), progs.begin() + N_ELITE);
        while (next_rules.size() < POP_SIZE) {
            auto [r, p] = crossover(rules[tournament(fit)], progs[tournament(fit)],
                                    rules[tournament(fit)], progs[tournament(fit)]);
            mutate(r, p);
            next_rules.push_back(std::move(r));
            next_progs.push_back(std::move(p));
        }

        std::uniform_int_distribution<> idx(N_ELITE, POP_SIZE - 1);
        for (std::size_t i = 0; i < N_IMM; ++i) {
            std::size_t j = idx(rng);
            next_rules[j] = random_rule();
            next_progs[j] = random_prog();
        }

        rules.swap(next_rules);
        progs.swap(next_progs);
    }

    std::ofstream csv("fitness_curve.csv");
    csv << "gen,best,mean\n";
    for (int g = 0; g < N_GENERATIONS; ++g)
        csv << (g+1) << "," << best_curve[g] << "," << mean_curve[g] << "\n";
    std::cout << "GA finished.\n";
    return 0;
}