#include <data/durations.h>
#include <data/options.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <pi/frame.h>
#include <pi/mcts.h>

#include <util/print.h>
#include <util/random.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <exception>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// TODO duration writing is probably not correct

static_assert(Options::calc && Options::chance && !Options::log);

bool terminated = false;
bool suspended = false;
constexpr size_t max_teams = SampleTeams::teams.size();

struct GameBuffer {

  std::vector<Frame> frames{};

  void add(const Frame &frame) { frames.emplace_back(frame); }

  void score(float score) {
    for (auto &frame : frames) {
      frame.score = score;
    }
  }

  void clear() { frames = {}; }

  auto fill(auto &container, auto &counter, auto max_size) {
    size_t added = 0;
    for (const auto &frame : frames) {
      if (counter >= max_size) {
        return added;
      }
      container[counter++] = frame;
      ++added;
    }
    return added;
  }
};

// worker local thread buffer that is filled before writing to disk
constexpr size_t thread_buffer_size = 1 << 12;
using ThreadFrameBuffer = std::array<Frame, thread_buffer_size>;
using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

void generate(int fd, std::atomic<size_t> *write_index,
              std::atomic<size_t> *frame_count, size_t global_buffer_size,
              std::chrono::milliseconds dur, uint64_t seed) {

  auto buffer_raw = new ThreadFrameBuffer{};
  auto &buffer = *buffer_raw;
  size_t buffer_size = 0;
  GameBuffer game_buffer{};
  prng device{seed};

  const auto write_thread_buffer_and_check_if_terminated = [&]() {
    const auto start_index = write_index->fetch_add(buffer_size);
    if (start_index >= global_buffer_size) {
      std::cout << "Terminating because global buffer size reached before write"
                << std::endl;
      terminated = true;
      return;
    }
    const auto end_index =
        std::min(start_index + buffer_size, global_buffer_size);
    std::cout << "writing to [" << start_index << ", " << end_index << ")."
              << std::endl;
    const auto p =
        pwrite(fd, buffer.data(), (end_index - start_index) * sizeof(Frame),
               start_index * sizeof(Frame));
    if (p != ((end_index - start_index) * sizeof(Frame))) {
      std::cerr << "Failed to write all of thread buffer." << std::endl;
    }
    buffer_size = 0;
    if (end_index >= global_buffer_size) {
      std::cout << "Terminating because global buffer size reached after write"
                << std::endl;
      terminated = true;
    }
    return;
  };

  // fat tail favoring smaller index (better teams)
  const auto random_index = [&device](size_t n) {
    size_t index = sqrt(device.random_int(n * n));
    return n - index;
  };

  const auto get_new_node = [](auto &unique_node, auto i1, auto i2,
                               const auto &obs) {
    auto *child = (*unique_node)(i1, i2, obs);
    if (!child) {
      unique_node = std::make_unique<Node>();
    } else {
      auto unique_child = unique_node->release_child(i1, i2, obs);
      unique_node.swap(unique_child);
    }
  };

  // think longer when more mons? Then frames need metadata
  const auto think_time = [&]() { return dur; };

  const auto prune_low_probs = [](auto &p) {
    const double t = 1.0d / p.size();
    double sum = 0;
    for (auto &x : p) {
      if (x < t) {
        x = 0;
      }
      sum += x;
    }
    if (sum == 0) {
      p[0] = 1;
      return;
    }
    for (auto &x : p) {
      x / sum;
    }
  };

  while (true) {

    const auto p1 = SampleTeams::teams[random_index(max_teams)];
    const auto p2 = SampleTeams::teams[random_index(max_teams)];
    auto battle = Init::battle(p1, p2);
    pkmn_gen1_chance_durations durations{};
    pkmn_gen1_battle_options options{};
    auto result = Init::update(battle, 0, 0, options);

    MCTS search{};
    MonteCarlo::Model mcm{device.uniform_64()};
    auto node = std::make_unique<Node>();

    try {

      while (!pkmn_result_type(result)) {

        // we need checks on a turn by turn basis, game by game is too slow.
        while (suspended) {
          sleep(1);
        }
        if (terminated) {
          write_thread_buffer_and_check_if_terminated();
          return;
        }

        const auto [p1_choices, p2_choices] = Init::choices(battle, result);
        if (p1_choices.size() == 1 && p2_choices.size() == 1) {
          result = Init::update(battle, p1_choices[0], p2_choices[0], options);
        } else {

          MonteCarlo::Input input{battle, durations, result};
          auto output = search.run(think_time(), *node.get(), input, mcm);
          Frame frame{&battle, &durations, result, output.average_value,
                      output.iterations};
          game_buffer.add(frame);

          prune_low_probs(output.p1);
          prune_low_probs(output.p2);
          const auto i1 = device.sample_pdf(output.p1);
          const auto i2 = device.sample_pdf(output.p2);
          const auto c1 = p1_choices[i1];
          const auto c2 = p2_choices[i2];
          result = Init::update(battle, c1, c2, options);
          durations = *pkmn_gen1_battle_options_chance_durations(&options);
          const auto &obs = *reinterpret_cast<std::array<uint8_t, 16> *>(
              pkmn_gen1_battle_options_chance_actions(&options));
          // TODO check this is correct place to set durations

          get_new_node(node, i1, i2, obs);
        }
      }

    } catch (const std::exception &e) {
      std::cerr << "Caught some exception in the game loop" << std::endl;
      std::cerr << e.what() << std::endl;
      game_buffer.clear();
    }

    const auto added =
        game_buffer.fill(buffer, buffer_size, thread_buffer_size);
    std::cout << "added: " << added << std::endl;
    game_buffer.clear();

    // flush thread buffer
    if (buffer_size >= thread_buffer_size) {
      write_thread_buffer_and_check_if_terminated();
    }

    // checks if there is enough frames when including the thread buffers
    const auto fc = frame_count->fetch_add(added);
    if (fc + added >= global_buffer_size) {
      write_thread_buffer_and_check_if_terminated();
      return;
    }
  }
}

void handle_print(std::atomic<size_t> *frame_count) {
  size_t done = 0;
  int sec = 10;
  while (!terminated) {
    sleep(sec);
    const auto more = frame_count->load();
    std::cout << (more - done) / (float)sec << " samples/sec." << std::endl;
    done = more;
  }
}

void handle_suspend(int signal) {
  std::cout << (suspended ? "Resumed." : "Suspended.") << std::endl;
  suspended = !suspended;
}

void handle_terminate(int signal) {
  std::cout << "TERMINATED" << std::endl;
  suspended = false;
  terminated = true;
}

int allocate_buffer(std::string filename, size_t global_buffer_size) {
  std::stringstream sstream{};

  const size_t file_size = global_buffer_size * sizeof(Frame);
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
  size_t global_buffer_size = 1 << 20;
  uint64_t seed = 2934828342938;

  if (argc != 5) {
    std::cerr
        << "Description: Generates MCTS training games and saves to a buffer."
        << std::endl;

    std::cerr << "input: threads, search time (ms), max frames, seed"
              << std::endl;
    return 1;
  }

  std::signal(SIGINT, handle_terminate);
  std::signal(SIGTSTP, handle_suspend);

  threads = std::atoi(argv[1]);
  ms = std::atoi(argv[2]);
  global_buffer_size = std::atoi(argv[3]);
  seed = std::atoi(argv[4]);

  std::cout << "Input: " << ms << ' ' << threads << ' ' << global_buffer_size
            << ' ' << seed << std::endl;

  int buffer_fd = allocate_buffer("buffer", global_buffer_size);
  if (buffer_fd == -1) {
    return 1;
  }

  std::atomic<size_t> write_index{};
  std::atomic<size_t> frame_count{};
  prng device{seed};

  std::thread thread_pool[threads];
  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{
        &generate,          buffer_fd,          &write_index,
        &frame_count,       global_buffer_size, std::chrono::milliseconds{ms},
        device.uniform_64()};
  }
  std::thread print_thread{&handle_print, &frame_count};

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }
  print_thread.join();

  const size_t frames_generated =
      std::min(write_index.load(), global_buffer_size);
  std::cout << "Generated " << frames_generated << " training frames."
            << std::endl;
  if (ftruncate(buffer_fd, frames_generated * sizeof(Frame)) != 0) {
    std::cerr << "Failed to truncate final buffer." << std::endl;
  }

  return 0;
}