#include <battle/init.h>
#include <data/sample-teams.h>

int main() {

  // 1v1 with some sleep moves. That seems to be the cause;

  const auto set_a = SampleTeams::teams[0][0];
  const auto set_b = SampleTeams::teams[0][1];
  std::vector<SampleTeam::Set> p1{set_a};
  std::vector<SampleTeam::Set> p2{set_b};

  const auto battle = Init::battle(p1, p2, 123456234);
  return 0;
}
