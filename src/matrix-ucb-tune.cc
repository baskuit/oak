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

int main() {
    using MatrixUCBTypes = TreeBandit<MatrixUCB<MonteCarloModelAverage<BattleTypes, false, true>>>;
    using Exp3Types = TreeBandit<Exp3<MonteCarloModelAverage<BattleTypes, false, true>>>;

    using T = Clamped<SearchModel<MatrixUCBTypes, false, true, true, false, true>>;
    using U = Clamped<SearchModel<MatrixUCBTypes, false, true, true, false, false>>;
    using V = Clamped<SearchModel<Exp3Types, false, true, true, false, true>>;

    // Model type that treats search output as its inference
    using ModelBanditTypes = TreeBanditThreaded<Exp3Fat<MonteCarloModel<ModelBandit>>>;
    // Type list for multithreaded exp3 over ModelBandit state

    std::vector<W::Types::Model> models{};
    const size_t ms = 500;
    std::vector<float> cuct{.5, 1.0, 1.7, 2.0};
    std::vector<size_t> mc_avg{1, 2, 4, 8};
    std::vector<float> expl{0.05, 0.1, .2};

    prng seed_device{4012024};

    models.emplace_back(W::make_model<V>(V::Model{ms, prng{0}, {prng{0}, 1 << 0}, {0.1}}));
    models.emplace_back(W::make_model<V>(V::Model{ms, prng{0}, {prng{0}, 1 << 1}, {0.1}}));
    models.emplace_back(W::make_model<V>(V::Model{ms, prng{0}, {prng{0}, 1 << 2}, {0.1}}));

    for (const float c : cuct) {
        for (const size_t t : mc_avg) {
            for (const float e : expl) {
                models.emplace_back(W::make_model<U>(
                    U::Model{ms, prng{seed_device.uniform_64()}, {prng{seed_device.uniform_64()}, t}, {c, e}}));
            }
        }
    }

    models.emplace_back(W::make_model<U>(U::Model{ms, prng{0}, {prng{0}, 1 << 3}, {2.5, 0.1}}));
    models.emplace_back(W::make_model<U>(U::Model{ms, prng{0}, {prng{0}, 1 << 3}, {1.5, 0.1}}));

    const size_t threads = 8;
    const size_t vs_rounds = 1;
    ModelBanditTypes::PRNG device{0};
    ModelBanditTypes::State arena_state{&generator_function, models, vs_rounds};
    ModelBanditTypes::Model arena_model{1337};
    ModelBanditTypes::Search search{ModelBanditTypes::BanditAlgorithm{.10}, threads};
    ModelBanditTypes::MatrixNode node{};

    const size_t arena_search_iterations = 1 << 6;
    const size_t n_prints = 50;

    {
        // expand without having to play an unlogged game with random models
        const size_t n_models = models.size();
        search.expand(node.stats, n_models, n_models, {});
        node.expand(n_models, n_models);
    }

    for (int i{}; i < n_prints; ++i) {
        search.run_for_iterations(arena_search_iterations, device, arena_state, arena_model, node);
        node.stats.matrix.print();
    }

    return 0;
}