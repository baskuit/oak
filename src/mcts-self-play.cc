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
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// TODO duration writing is probably not correct

static_assert(Options::calc && Options::chance && !Options::log);

bool terminated = false;
bool suspended = false;
constexpr size_t max_teams = SampleTeams::teams.size();

struct Frame {
  std::array<uint8_t, Sizes::battle> battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
  float eval;
  float score;
  // float ms;
  uint32_t iter;

  Frame() = default;
  Frame(const pkmn_gen1_battle *const battle,
        const pkmn_gen1_chance_durations *const durations, pkmn_result result,
        float eval)
      : durations{*durations}, result{result}, eval{eval} {
    std::memcpy(this->battle.data(), battle, Sizes::battle);
  }
};

struct GameBuffer {

  std::vector<Frame> frames{};

  void add(const Frame &frame) { frames.emplace_back(frame); }

  void score(float score) {
    for (auto &frame : frames) {
      frame.score = score;
    }
  }

  void clear() { frames.clear(); }

  void fill(auto &container, auto &counter, auto max_size) {
    for (const auto &frame : frames) {
      container[counter++] = frame;
      if (counter >= max_size) {
        return;
      }
    }
  }
};

// worker local thread buffer that is filled before writing to disk
constexpr size_t max_buffer_size = 1 << 4;
using ThreadFrameBuffer = std::array<Frame, max_buffer_size>;
using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

void generate(int fd, std::atomic<int> *frame_counter, std::mutex *write_mutex,
              size_t max_frames, std::chrono::milliseconds dur, uint64_t seed) {

  auto buffer_raw = new ThreadFrameBuffer{};
  auto &buffer = *buffer_raw;
  size_t buffer_size = 0;
  GameBuffer game_buffer{};
  prng device{seed};

  const auto write_buffer_to_disk = [&]() {
    // std::unique_lock lock{*write_mutex};
    const auto start_index = frame_counter->fetch_add(buffer_size);
    std::cout << "writing to index " << start_index << std::endl;
    pwrite(fd, buffer.data(), buffer_size * sizeof(Frame), start_index);
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

  // think longer when more mons? Then frames need metadata
  const auto think_time = [&]() { return dur; };

  while (true) {

    const auto p1 = SampleTeams::teams[random_index(max_teams)];
    const auto p2 = SampleTeams::teams[random_index(max_teams)];
    auto battle = Init::battle(p1, p2);
    pkmn_gen1_chance_durations durations{};
    pkmn_gen1_battle_options options{};
    auto result = Init::update(battle, 0, 0, options);

    MCTS search{};
    MonteCarlo::Model mcm{device.uniform_64()};

    try {

      while (!pkmn_result_type(result)) {

        // we need checks on a turn by turn basis, game by game is too slow.
        while (suspended) {
          sleep(1);
        }
        if (terminated) {
          write_buffer_to_disk();
          return;
        }

        const auto [p1_choices, p2_choices] = Init::choices(battle, result);
        std::cout << "choice sizes: " << p1_choices.size() << ' '
                  << p2_choices.size() << std::endl;
        if (p1_choices.size() == 1 && p2_choices.size() == 1) {
          result = Init::update(battle, p1_choices[0], p2_choices[0], options);
        } else {

          Node node{};
          MonteCarlo::Input input{};
          input.battle = battle;
          input.durations =
              *pkmn_gen1_battle_options_chance_durations(&options);
          input.result = result;
          const auto output = search.run(think_time(), node, input, mcm);
          Frame frame{&battle, &durations, result, output.average_value};
          game_buffer.add(frame);

          const auto i1 = device.sample_pdf(output.p1);
          const auto i2 = device.sample_pdf(output.p2);
          const auto c1 = p1_choices[i1];
          const auto c2 = p2_choices[i2];

          // for (const auto x : output.p1) {
          //   std::cout << x << ' ';
          // }
          // std::cout << std::endl;
          // for (const auto x : output.p2) {
          //   std::cout << x << ' ';
          // }
          // std::cout << std::endl;
          // std::cout << "indexes: " << i1 << ' ' << i2 << std::endl;

          result = Init::update(battle, c1, c2, options);
        }

        std::cout << "Frame: " << game_buffer.frames.size() << std::endl;
      }

    } catch (...) {
      std::cerr << "Caught some exception in the game loop" << std::endl;
      game_buffer.clear();
    }

    game_buffer.fill(buffer, buffer_size, max_buffer_size);
    game_buffer.clear();
  }
}

void handle_print() {
  while (!terminated) {
    sleep(1);
  }
}

void handle_suspend(int signal) {
  std::cout << (suspended ? "SUPSPENDED" : "RESUMED") << std::endl;
  suspended = !suspended;
}

void handle_terminate(int signal) {
  std::cout << "TERMINATED" << std::endl;
  terminated = true;
}

int allocate_buffer(std::string filename, size_t max_frames) {
  std::stringstream sstream{};

  const size_t file_size = max_frames * sizeof(Frame);
  int fd;
  try {
    fd = open(filename.data(), O_WRONLY | O_CREAT, 0644);
    if (ftruncate(fd, file_size) != 0) {
      return -1;
    }
  } catch (...) {
    std::cerr << "Cound not allocate " << file_size << " bytes on disk"
              << std::endl;
    return -1;
  }
  std::cout << "Able to open file" << std::endl;
  return fd;
}

int main(int argc, char **argv) {

  size_t threads = 1;
  size_t ms = 100;
  size_t max_frames = 1 << 20;
  uint64_t seed = 2934828342938;

  // if (argc != 5) {
  //   std::cerr << "desc.: Generates MCTS training games and saves to a
  //   buffer."
  //             << std::endl;

  //   std::cerr << "input: threads, search time (ms), max frames, seed"
  //             << std::endl;
  //   return 1;
  // }

  std::cout << "frame size: " << sizeof(Frame) << std::endl;

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  // threads = std::atoi(argv[1]);
  // ms = std::atoi(argv[2]);
  // max_frames = std::atoi(argv[3]);
  // seed = std::atoi(argv[4]);

  std::cout << "Input: " << ms << ' ' << threads << ' ' << max_frames << ' '
            << seed << std::endl;

  int buffer_fd = allocate_buffer("buffer", max_frames);
  if (buffer_fd == -1) {
    return 1;
  }

  std::atomic<int> frame_counter{};
  std::mutex write_mutex{};
  prng device{seed};

  std::thread thread_pool[threads];
  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{
        &generate,          buffer_fd,  &frame_counter,
        &write_mutex,       max_frames, std::chrono::milliseconds{ms},
        device.uniform_64()};
  }
  std::thread print_thread{&handle_print};

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }
  print_thread.join();

  // resize file to actual frame_counter size
  const size_t actual_size = frame_counter.load() * sizeof(Frame);
  if (ftruncate(buffer_fd, actual_size) != 0) {
    std::cerr << "Failed to truncate final buffer to " << actual_size
              << " bytes." << std::endl;
  }

  std::cout << "generated " << frame_counter.load() << " training frames"
            << std::endl;

  return 0;
}