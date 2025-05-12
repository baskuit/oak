#include <array>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <string>

// clang-format OFF

#include <string>

enum Mon {
    KO,
    G1,
    H1,
    A1,
    G2,
    H2,
    A2,
    G3,
    H3,
    A3,
    G4,
};

std::string mon_to_string(const Mon mon) {
    const std::string names[11]{
        "KO", "G1", "H1", "A1", "G2", "H2", "A2", "G3", "H3", "A3", "G4"
    };
    return names[static_cast<int>(mon)];
}

constexpr int speed(const Mon mon) {
    return 2 - ((static_cast<int>(mon) - 1) % 3); 
}
static_assert(speed(G4) == 2);

Mon apply_damage(Mon m) {
    if (m == G1 || m == H1 || m == A1) {
        return KO;
    }
    return static_cast<Mon>(static_cast<int>(m) - 3);
}

constexpr Mon mons[11] = 
{KO, G1, H1, A1, G2, H2, A2, G3, H3, A3, G4};

using Bench = std::array<int, 11>;

auto generate_benches() {
    std::array<std::vector<Bench>, 6> benches_per_count{};
    for (int ko = 0, sum0 = ko; ko < 6; ++ko, sum0 = ko)
    for (int g1 = 0, sum1 = sum0 + g1; g1 < 6 && sum1 < 6; ++g1, sum1 = sum0 + g1)
    for (int g2 = 0, sum2 = sum1 + g2; g2 < 6 && sum2 < 6; ++g2, sum2 = sum1 + g2)
    for (int g3 = 0, sum3 = sum2 + g3; g3 < 6 && sum3 < 6; ++g3, sum3 = sum2 + g3)
    for (int g4 = 0, sum4 = sum3 + g4; g4 < 6 && sum4 < 6; ++g4, sum4 = sum3 + g4)
    for (int h1 = 0, sum5 = sum4 + h1; h1 < 6 && sum5 < 6; ++h1, sum5 = sum4 + h1)
    for (int h2 = 0, sum6 = sum5 + h2; h2 < 6 && sum6 < 6; ++h2, sum6 = sum5 + h2)
    for (int h3 = 0, sum7 = sum6 + h3; h3 < 6 && sum7 < 6; ++h3, sum7 = sum6 + h3)
    for (int a1 = 0, sum8 = sum7 + a1; a1 < 6 && sum8 < 6; ++a1, sum8 = sum7 + a1)
    for (int a2 = 0, sum9 = sum8 + a2; a2 < 6 && sum9 < 6; ++a2, sum9 = sum8 + a2)
    for (int a3 = 0, sum10 = sum9 + a3; a3 < 6 && sum10 < 6; ++a3, sum10 = sum9 + a3) {
        if (sum10 == 5) {
            benches_per_count[5 - ko].push_back({ko, g1, g2, g3, g4, h1, h2, h3, a1, a2, a3});
        }
    }
    return benches_per_count;
}

const auto benches_per_count = generate_benches();

struct Side {
    Mon active;
    Bench bench;

    bool operator==(const Side&) const = default;

    void print() const {
            std::cout << mon_to_string(active) << " : ";
        for (int i = 1; i < 11; ++i) {
            for (int n = 0; n < bench[i]; ++n) {
                std::cout << mon_to_string(static_cast<Mon>(i)) << ' ';
            }
        }
    }
};

using Matchup = std::pair<Side, Side>;
using Value = double;
using Table = std::unordered_map<Matchup, Value>;

void get_value(const Matchup& mu) {
    const auto &p1 = mu.first;
    const auto &p2 = mu.second;


    std::array<std::array<Value, 11>, 11> p1_values;
    std::array<std::array<Value, 11>, 11> p2_values;

    

}


auto generate_sides () {
    std::vector<Side> sides{};
    for (int n_bench = 0; n_bench < 6; ++n_bench) {
        for (int i = 0; i < 11; ++i) {
            const auto active = static_cast<Mon>(i);

            for (const auto bench : benches_per_count[n_bench]) {
                sides.push_back(Side{active, bench});
            }
        }
    }
    return sides;
}

const auto sides = generate_sides();

void solve() {


    const auto n = sides.size();
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j <= n; ++j) {
            const auto &s1 = sides[i];
            const auto &s2 = sides[j];

            const bool any_ko = (s1.active == KO) || (s2.active == KO);

            const bool tie_matters = (static_cast<int>(s1.active) < 4) && (static_cast<int>(s2.active) < 4); 

            std::array<std::array<Value, 11>, 11> p1_matrix;
            std::array<std::array<Value, 11>, 11> p2_matrix;
            for (int i = 0; i < 10; ++i) {
                for (int j = 0; j < 10; ++j) {
                    p1_matrix[i][j] == -1;
                    p2_matrix[i][j] == -1;
                }
            }
        }
    }


}

int main () {


    for (int i = 0; i < 59; ++i) {
        sides[i].print();
        std::cout << std::endl;
    }


}