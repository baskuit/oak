#include <data/durations.h>
#include <data/options.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <pi/mcts.h>

#include <util/print.h>
#include <util/random.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

static_assert(Options::calc && Options::chance && !Options::log);

bool terminated = false;
bool suspended = false;
constexpr size_t max_teams = 500;
// TODO some way of fat tail sampling the teams
// Also probably a way to adjust think time

struct Frame {
  uint8_t battle[Sizes::battle];
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  float eval;
  float score;

  Frame() = default;
  Frame(const pkmn_gen1_battle *const battle,
        const pkmn_gen1_chance_durations *const durations, pkmn_result result,
        float eval)
      : durations{*durations}, result{result}, eval{eval} {
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
constexpr size_t max_buffer_size = 1 << 16;
using ThreadFrameBuffer = std::array<Frame, max_buffer_size>;
using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

void generate(std::atomic<int> *frame_counter, std::mutex *write_mutex,
              size_t max_frames, std::chrono::milliseconds dur, uint64_t seed) {

  ThreadFrameBuffer buffer{};
  size_t buffer_size = 0;
  GameBuffer game_buffer{};
  prng device{seed};

  const auto write_buffer_to_disk = [&]() {
    std::unique_lock lock{*write_mutex};
    buffer_size = 0;
    return;
  };

  // fat tail favoring smaller index (better teams)
  const auto random_index = [&device](size_t n) {
    size_t index = sqrt(device.random_int(n * n));
    return n - index;
  };

  // probably a good idea to reuse the tree from last turn
  const auto keep_node = [](const auto node, const auto battle,
                            const auto obs) { return Node{}; };

  const auto think_time = [&]() { return dur; };

  while (true) {

    try {
      const auto p1 = SampleTeams::teams[random_index(max_teams)];
      const auto p2 = SampleTeams::teams[random_index(max_teams)];
      auto battle = Init::battle(p1, p2);
      pkmn_gen1_chance_durations durations{};
      pkmn_gen1_battle_options options{};
      auto result = Init::update(battle, 0, 0, options);
      bool valid = true;

      MCTS search{};
      MonteCarlo::Model mcm{device.uniform_64()};

      while (!pkmn_result_type(result)) {

        // we need checks on a turn by turn basis, game by game is too slow.
        while (suspended) {
          sleep(1);
        }
        if (terminated ||
            (frame_counter->load() + game_buffer.index > max_frames)) {
          write_buffer_to_disk();
          return;
        }

        const auto [p1_choices, p2_choices] = Init::choices(battle, result);
        if (p1_choices.size() == 1 && p2_choices.size() == 1) {
          result = Init::update(battle, p1_choices[0], p2_choices[0], options);
        } else {

          Node node{};

          MonteCarlo::Input input{};
          auto output = search.run(think_time(), node, input, mcm);
          Frame frame{&battle, &durations, result, output.average_value};
          if (game_buffer.add(frame) == false) {
            valid = false;
            break;
          }

          const auto c1 = device.sample_pdf(output.p1);
          const auto c2 = device.sample_pdf(output.p2);
          result = Init::update(battle, c1, c2, options);
        }
      } // game loop

      if (valid) {

        game_buffer.score(Init::score(result));

        const auto current_frames = frame_counter->fetch_add(game_buffer.index);
        if (current_frames >= max_frames) {
          write_buffer_to_disk();
          return;
        } else {
          for (auto i = 0; i < game_buffer.index; ++i) {
            buffer[buffer_size++] = game_buffer.frames[i];
            if (buffer_size >= max_buffer_size) {
              write_buffer_to_disk();
              break;
            }
          }
        }
      }

      game_buffer.clear();
    } catch (...) {
    }
  } // main while
}

void handle_print() {
  while (true) {
    sleep(1);
  }
}

void handle_suspend(int signal) {
  std::cout << "\nCaught signal: " << signal << std::endl;
  suspended = !suspended; // TODO lol
}

void handle_terminate(int signal) {
  std::cout << "\nCaught signal: " << signal << std::endl;
  terminated = true;
}

int main(int argc, char **argv) {

  size_t threads = 32;
  size_t ms = 100;
  size_t max_frames = 1 << 26;
  uint64_t seed = 2934828342938;

  if (argc != 5) {
    std::cerr << "desc.: Generates MCTS training games and saves to a buffer."
              << std::endl;
    std::cerr << "input: threads, search time (ms), max frames, seed"
              << std::endl;
    return 1;
  }

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  threads = std::atoi(argv[1]);
  ms = std::atoi(argv[2]);
  max_frames = std::atoi(argv[3]);
  seed = std::atoi(argv[4]);

  std::cout << "Input: " << ms << ' ' << threads << ' ' << max_frames << ' '
            << seed << std::endl;

  std::atomic<int> frame_counter{};
  std::mutex write_mutex{};
  prng device{seed};

  std::thread thread_pool[threads];
  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{&generate,
                                 &frame_counter,
                                 &write_mutex,
                                 max_frames,
                                 std::chrono::milliseconds{ms},
                                 device.uniform_64()};
  }
  std::thread print_thread{&handle_print};

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }
  print_thread.join();

  // resize file to actual frame_counter size

  std::cout << "generated " << frame_counter.load() << " training frames"
            << std::endl;

  return 0;
}