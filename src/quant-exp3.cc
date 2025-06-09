#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <algorithm>

constexpr int NUM_ARMS = 5;
constexpr int FP_SHIFT = 16;
constexpr uint32_t FP_ONE = 1 << FP_SHIFT;
constexpr uint32_t GAMMA = (uint32_t)(0.07 * FP_ONE);  // 0.07 in Q16.16
constexpr uint32_t GAMMA_OVER_K = GAMMA / NUM_ARMS;

// Approximate exp(x) in Q16.16 using 1 + x + x²/2 + x³/6
uint32_t approx_exp_q16(uint32_t x_q16) {
    uint32_t x2 = (x_q16 * x_q16) >> FP_SHIFT;
    uint32_t x3 = (x2 * x_q16) >> FP_SHIFT;
    return FP_ONE + x_q16 + (x2 >> 1) + (x3 / 6);
}

// Sample an arm using fixed-point probabilities
int sample_arm(const std::vector<uint32_t>& probs_q16, std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, FP_ONE - 1);
    uint32_t r = dist(rng);
    uint32_t acc = 0;
    for (int i = 0; i < probs_q16.size(); ++i) {
        acc += probs_q16[i];
        if (r < acc)
            return i;
    }
    return probs_q16.size() - 1;
}

int main() {
    std::vector<uint16_t> gains(NUM_ARMS, 0);         // Quantized gain
    std::vector<uint32_t> weights(NUM_ARMS, FP_ONE);  // exp(gain)
    std::vector<uint32_t> probs(NUM_ARMS, FP_ONE / NUM_ARMS); // Probabilities

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> reward_gen(0, 100); // Simulated reward in [0, 100]

    for (int t = 1; t <= 1000; ++t) {
        // Compute weights as exp(gain * gamma / k)
        uint64_t weight_sum = 0;
        for (int i = 0; i < NUM_ARMS; ++i) {
            uint32_t gain_scaled = (gains[i] * GAMMA_OVER_K); // Q16.16
            weights[i] = approx_exp_q16(gain_scaled);
            weight_sum += weights[i];
        }

        // Compute probabilities in Q16.16
        for (int i = 0; i < NUM_ARMS; ++i) {
            uint32_t prob_exploit = (uint32_t)(((uint64_t)weights[i] << FP_SHIFT) / weight_sum); // Q16.16
            uint32_t prob_explore = FP_ONE / NUM_ARMS; // Uniform
            // (1 - gamma) * exploit + gamma * explore
            probs[i] = ((FP_ONE - GAMMA) * prob_exploit + GAMMA * prob_explore) >> FP_SHIFT;
        }

        // Choose an arm
        int chosen = sample_arm(probs, rng);

        // Simulate reward in [0, 1], scaled to Q16.16
        uint32_t reward_q16 = (reward_gen(rng) * FP_ONE) / 100;

        // Estimated reward: reward / prob (both Q16.16)
        uint32_t estimated = ((uint64_t)reward_q16 << FP_SHIFT) / probs[chosen]; // Q16.16

        // Convert estimated reward to scaled integer gain and clip
        uint32_t gain_update = estimated >> (FP_SHIFT - 4); // scale down to uint16_t range
        gains[chosen] = std::min<uint32_t>(65535, gains[chosen] + gain_update);

        // Debug print
        std::cout << "Round " << t << ": chose arm " << chosen
                  << ", reward=" << (reward_q16 >> 16)
                  << ", prob=" << (probs[chosen] >> 8) << "/256\n";
    }

    return 0;
}
