#pragma once

#include "ucb.h"
// TODO template, right now its ~1.1 Gb in the main table with 2^24 entries
// overflow can have up to 2^40 entries, but lets say 2^22 so its less than 2Gb
// altogether;
class TT {
  using namespace UCB;
  using OverflowHandle = uint32_t;

  RootUCBNode root_node;
  std::array<UCBNode, 1 << 24> main_table;
  std::array<UCBNode, 1 << 22> overflow_table;
  OverflowHandle overflow{};

  // linearly scans overflow node handle path
  UCBNode *find_node_overflow(const uint64_t hash,
                              const OverflowHandle handle) noexcept {
    UCBNode *current = overflow_table.data() + (handle % (1 << 22));
    return nullptr;
  }

public:
  // allocates if it does not find
  // returns nullptr if and only iff it could not allocate
  UCBNode *find_node(const uint64_t hash) noexcept {
    auto &first = main_table[hash >> 40];
    const bool match_collision = std::memcmp(first.collision, &hash, 5) == 0;
    return match_collision ? &first : find_node_overflow(hash, first.overflow);
  }
};

class TTTest {
  // less than 2Gb
  static_assert(sizeof(TT) < (1 << 31));
};