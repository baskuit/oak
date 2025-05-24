#include <data/durations.h>
#include <data/options.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <pi/mcts.h>

#include <util/print.h>
#include <util/random.h>

#include <atomic>
#include <iostream>
#include <mutex>
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
  Frame(const pkmn_gen1_battle *const battle, pkmn_result result, float eval)
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
  prng device{seed};

  const auto exit = []() { return; };

  while (true) {

    auto current_index = index->load();

    const auto p1 = SampleTeams::teams[device.random_int(100)];
    const auto p2 = SampleTeams::teams[device.random_int(100)];
    auto battle = Init::battle(p1, p2);
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