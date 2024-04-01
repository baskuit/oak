// #include <pinyon.hh>

#include "../include/battle.hh"
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
    using MatrixUCBTypes = TreeBandit<MatrixUCB<MonteCarloModelAverage<BattleTypes>>>;
    using Exp3Types = TreeBandit<Exp3<MonteCarloModelAverage<BattleTypes>>>;

    // Exp3 search types on a solved random tree
    using T = SearchModel<MatrixUCBTypes, false, true, true, false>;
    using U = SearchModel<Exp3Types, false, true, true, false>;

    // Model type that treats search output as its inference
    using ModelBanditTypes = TreeBanditThreaded<Exp3Fat<MonteCarloModel<ModelBandit>>>;
    // Type list for multithreaded exp3 over ModelBandit state

    std::vector<W::Types::Model> models{};
    const size_t ms = 500;
    models.emplace_back(W::make_model<T>(T::Model{ms, prng{0}, {prng{0}, 1 << 3}, {2.0, 0.1}}));
    models.emplace_back(W::make_model<T>(T::Model{ms, prng{0}, {prng{0}, 1 << 3}, {2.5, 0.1}}));
    models.emplace_back(W::make_model<T>(T::Model{ms, prng{0}, {prng{0}, 1 << 3}, {1.5, 0.1}}));

    const size_t threads = 4;
    const size_t vs_rounds = 1;
    ModelBanditTypes::PRNG device{0};
    ModelBanditTypes::State arena_state{&generator_function, models, vs_rounds};
    ModelBanditTypes::Model arena_model{1337};
    ModelBanditTypes::Search search{ModelBanditTypes::BanditAlgorithm{.10}, threads};
    ModelBanditTypes::MatrixNode node{};

    const size_t arena_search_iterations = 1 << 3;
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