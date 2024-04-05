#include "../include/battle.hh"
#include "../include/clamp.hh"
#include "../include/mc-average.hh"
#include "../include/sides.hh"

using BattleTypes = Battle<0, 3, ChanceObs, float, float>;

W::Types::State generator_function(const W::Types::Seed seed) {
    prng device{seed};

    const int r = device.random_int(n_sides - 1);
    const int c = device.random_int(n_sides - 1);
    BattleTypes::State state{sides[r + 1], sides[c + 1]};
    state.randomize_transition(device);

    return W::make_state<BattleTypes>(state);
}

template <typename Types>
void print_matrix(typename Types::MatrixNode *node) {
    while (true) {
        sleep(300);
        node->stats.matrix.print();
        std::cout << std::endl;
    }
}

int main() {
    using MatrixUCBTypes = TreeBandit<MatrixUCB<MonteCarloModelAverage<BattleTypes, false, true>>>;
    using Exp3Types = TreeBandit<UCB<MonteCarloModelAverage<BattleTypes, false, true>>>;
    using UCBTypes = TreeBandit<UCB<MonteCarloModelAverage<BattleTypes, false, true>>>;

    // iter, tree bandit, policy ,value, empirical
    using M = Clamped<SearchModel<MatrixUCBTypes, false, true, true, false, true>>;
    using U = Clamped<SearchModel<UCBTypes, false, true, true, false, true>>;
    using E = Clamped<SearchModel<Exp3Types, false, true, true, false, false>>;

    // Model type that treats search output as its inference
    using ModelBanditTypes = TreeBanditThreaded<Exp3Fat<MonteCarloModel<ModelBandit>>>;
    // Type list for multithreaded exp3 over ModelBandit state

    std::vector<W::Types::Model> models{};
    const size_t ms = 500;
    std::vector<float> cuct{.5, 1.0, 1.7, 2.0};
    std::vector<size_t> mc_avg{1, 2, 4, 8};
    std::vector<float> expl{0.05, 0.1, .2};

    prng seed_device{4753498573498};

    {
        models.emplace_back(
            W::make_model<U>(U::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, 1}, {2}}));
        // models.emplace_back(W::make_model<T>(
        //     T::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, 1}, {1, .1}}));
        // models.emplace_back(W::make_model<T>(
        //     T::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, 1}, {1.5, .1}}));
        models.emplace_back(
            W::make_model<E>(E::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, 1}, {.1}}));
        // models.emplace_back(
        //     W::make_model<U>(U::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, 1},
        //     {.02}}));
    };

    // models.emplace_back(W::make_model<V>(V::Model{ms, prng{0}, {prng{0}, 1 << 0}, {0.1}}));
    // models.emplace_back(W::make_model<V>(V::Model{ms, prng{0}, {prng{0}, 1 << 1}, {0.1}}));
    // models.emplace_back(W::make_model<V>(V::Model{ms, prng{0}, {prng{0}, 1 << 2}, {0.1}}));

    // for (const float c : cuct) {
    //     for (const size_t t : mc_avg) {
    //         for (const float e : expl) {
    //             models.emplace_back(W::make_model<U>(
    //                 U::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, t}, {c, e}}));
    //         }
    //     }
    // }

    const size_t threads = 4;
    const size_t vs_rounds = 1;
    ModelBanditTypes::PRNG device{seed_device.uniform_64()};
    ModelBanditTypes::State arena_state{&generator_function, models, vs_rounds};
    ModelBanditTypes::Model arena_model{1337};
    ModelBanditTypes::Search search{ModelBanditTypes::BanditAlgorithm{.10}, threads};
    ModelBanditTypes::MatrixNode node{};

    const size_t arena_search_iterations = 1 << 14;

    {
        // expand without having to play an unlogged game with random models
        const size_t n_models = models.size();
        search.expand(node.stats, n_models, n_models, {});
        node.expand(n_models, n_models);
    }

    std::thread print_thread{&print_matrix<ModelBanditTypes>, &node};

    search.run_for_iterations(arena_search_iterations, device, arena_state, arena_model, node);

    return 0;
}