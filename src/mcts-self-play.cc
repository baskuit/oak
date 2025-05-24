#include <data/durations.h>
#include <data/options.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <util/print.h>
#include <util/random.h>

#include <atomic>
#include <iostream>
#include <thread>

static_assert(Options::calc && Options::chance && !Options::log);

bool terminated = false;
bool suspended = false;

struct Frame {
  uint8_t battle[Sizes::battle];
  pkmn_result result;
  float eval;
  float score;

  Frame() = default;
  Frame(const pkmn_gen1_battle const *battle, pkmn_result result, float eval)
      : result{result}, eval{eval} {
    std::memcpy(this->battle, battle, Sizes::battle);
  }
};

struct GameBuffer {

  Frame frames[1012]; // probably an upper bound
  size_t index;

  bool add(const Frame &frame) {
    if (index >= 1012) {
      return false;
    }
    frames[index] = frame;
    ++index;
    return true;
  }

  void score(float score) {
    for (auto i = 0; i < index; ++i) {
      frames[i].score = score;
    }
  }

  void clear() { index = 0; }
};

// worker local thread buffer that is filled before writing to disk
using FrameBuffer = std::array<Frame, 1 << 16>;

template <typename Team, typename Dur>
void generate(std::atomic<int> *index, size_t max, Dur dur, uint64_t seed,
              size_t *n, size_t *score) {

  GameBuffer game_buffer{};

  const auto exit = []() { return; };

  while (true) {

    auto current_index = index.load();

    auto battle = Init::battle(x, y);
    pkmn_gen1_chance_durations durations{};
    pkmn_gen1_battle_options options{};
    auto result = Init::update(battle, 0, 0, options);

    MCTS search{};
    MonteCarlo::Model mcm{seed};

    while (!pkmn_result_type(result)) {
      while (suspended) {
        sleep(1);
      }
      if (terminated) {

        exit();
        return;
      }
    }
  }

  while (index->fetch_add(1) < max) {

    const auto half = [dur](auto x, auto y, auto seed) -> float {
      auto battle = Init::battle(x, y);
      pkmn_gen1_chance_durations durations{};
      pkmn_gen1_battle_options options{};
      auto result = Init::update(battle, 0, 0, options);

      MCTS search{};
      MonteCarlo::Model mcm{seed};

      std::vector<float> eval_values{};

      while (!pkmn_result_type(result)) {

        const auto [choices1, choices2] = Init::choices(battle, result);

        auto i = 0;
        if (choices1.size() * choices2.size() > 1) {
          using Node = Tree::Node<Exp3::JointBanditData<.03f, false>,
                                  std::array<uint8_t, 16>>;
          Node node{};
          MonteCarlo::Input input1{battle, durations, result};
          auto output1 =
              search.run<MCTS::Options<true, 3, 3, 1>>(dur, node, input1, mcm);
          i = mcm.device.sample_pdf(output1.p1);
          eval_values.push_back(1 - output1.average_value);
        }

        result = Init::update(battle, choices1[i], choices2[j], options);
        durations = *pkmn_gen1_battle_options_chance_durations(&options);
      }
      const auto v = eval_values.size();
      for (auto vi = 0; vi < v; ++vi) {
        std::cout << mcts_value_for_eval_side[vi] << " ~ " << eval_values[vi]
                  << '\t';
      }
      const auto score = Init::score(result);
      std::cout << "~ " << 1 - score << std::endl;

      return score;
    };

    prng device{seed};

    const auto p1 = SampleTeams::teams[device.random_int(100)];
    const auto p2 = SampleTeams::teams[device.random_int(100)];
  }
}

void print_score(bool *flag, size_t *n, size_t *score) {
  while (!*flag) {
    float n_ = static_cast<float>(*n) + 1.0f * (*n == 0);
    float avg = static_cast<float>(*score) / 2 / n_;
    std::cout << "n: " << *n << " avg: " << avg << std::endl;
    sleep(10);
  }
}

template <size_t frame_size> struct Worker {
  Frame frames[frame_size];
  size_t current_frame_index;
  std::mutex *mtx;

  void flush_frames(const char *stream) {
    std::unique_lock lock{*mtx};
    // write to stream
    current_frame_index = 0;
  }
};

int main(int argc, char **argv) {

  using Team = std::array<Init::Set, 6>;

  size_t threads = 32;
  size_t ms = 100;
  size_t max_frames = 1 << 26;
  uint64_t seed = 2934828342938;

  if (argc != 5) {
    std::cerr
        << "Generates MCTS training games with MCTS and saves to a buffer."
        << std::endl;
    std::cerr << "Input: threads, search time (ms), max frames, seed"
              << std::endl;
    return 1;
  }

  threads = std::atoi(argv[1]);
  ms = std::atoi(argv[2]);
  max_frames = std::atoi(argv[3]);
  seed = std::atoi(argv[4]);

  std::cout << "Input: " << ms << ' ' << threads << ' ' << max_frames << ' '
            << seed << std::endl;

  std::atomic<int> index{};
  prng device{seed};

  size_t n = 0;
  size_t score = 0;

  std::thread thread_pool[threads];

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{&generate<Team, std::chrono::milliseconds>,
                                 &index,
                                 max_frames,
                                 std::chrono::milliseconds{ms},
                                 device.uniform_64(),
                                 &n,
                                 &score};
  }

  bool flag = false;

  std::thread print_thread{&print_score, &flag, &n, &score};

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }

  flag = true;
  print_thread.join();

  return 0;
}