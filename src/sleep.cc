#include <battle/init.h>
#include <data/sample-teams.h>

#include <array>
#include <iostream>
#include <vector>

void get_choices(const auto &battle, auto &p1_choices, auto &p2_choices) {}

int main() {

  // 1v1 with some sleep moves. That seems to be the cause;

  const auto set_a = SampleTeams::teams[0][0];
  const auto set_b = SampleTeams::teams[0][1];
  std::vector<SampleTeams::Set> p1{set_a};
  std::vector<SampleTeams::Set> p2{set_b};

  std::array<pkmn_choice, 9> p1_choices{};
  std::array<pkmn_choice, 9> p2_choices{};

  const auto battle = Init::battle(p1, p2, 123456234);
  return 0;
}
