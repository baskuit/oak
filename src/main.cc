#include <iostream>
#include <mutex>

#include <types/random.h>
#include <types/vector.h>

#include <battle/battle.h>
#include <battle/chance.h>
#include <battle/data/data.h>
#include <battle/debug-log.h>
#include <battle/sides.h>

// ad hoc side info for a battle
struct Set {
  Data::Species species;
  std::vector<Data::Moves> moves;
};

namespace SampleTeams {
using Data::Moves;
using Data::Species;
using enum Moves;
std::array<Set, 6> team0 = {
    Set{Species::Jynx, {Blizzard, LovelyKiss, Psychic, Rest}},
    {Species::Chansey, {IceBeam, Sing, SoftBoiled, Thunderbolt}},
    {Species::Cloyster, {Blizzard, Clamp, Explosion, HyperBeam}},
    {Species::Rhydon, {BodySlam, Earthquake, RockSlide, Substitute}},
    {Species::Starmie, {Blizzard, Recover, Thunderbolt, ThunderWave}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};
std::array<Set, 6> team1 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, Psychic, SleepPowder, StunSpore}},
    {Species::Lapras, {Blizzard, HyperBeam, Sing, Thunderbolt}},
    {Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<Set, 6> team2 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Counter, IceBeam, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, Psychic, SleepPowder, StunSpore}},
    {Species::Lapras, {Blizzard, HyperBeam, Sing, Thunderbolt}},
    {Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<Set, 6> team3 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, HyperBeam, Psychic, SleepPowder}},
    {Species::Snorlax, {BodySlam, Earthquake, Reflect, Rest}},
    {Species::Starmie, {Blizzard, Recover, Thunderbolt, ThunderWave}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<Set, 6> team4 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {DoubleEdge, Explosion, Psychic, SleepPowder}},
    {Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}},
    {Species::Zapdos, {Agility, DrillPeck, Thunderbolt, ThunderWave}}};

std::array<Set, 6> team5 = {
    Set{Species::Starmie, {Blizzard, Psychic, Recover, ThunderWave}},
    {Species::Chansey, {IceBeam, SoftBoiled, Thunderbolt, ThunderWave}},
    {Species::Exeggutor, {Explosion, Psychic, SleepPowder, StunSpore}},
    {Species::Rhydon, {BodySlam, Earthquake, RockSlide, Substitute}},
    {Species::Snorlax, {BodySlam, Earthquake, Reflect, Rest}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<Set, 6> team6 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, Psychic, SleepPowder, StunSpore}},
    {Species::Lapras, {Blizzard, BodySlam, Sing, Thunderbolt}},
    {Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<Set, 6> team7 = {
    Set{Species::Starmie, {Blizzard, Psychic, Recover, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, HyperBeam, Psychic, SleepPowder}},
    {Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}},
    {Species::Zapdos, {Agility, DrillPeck, Thunderbolt, ThunderWave}}};

std::array<Set, 6> team8 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, Psychic, SleepPowder, StunSpore}},
    {Species::Lapras, {Blizzard, HyperBeam, Sing, Thunderbolt}},
    {Species::Snorlax, {BodySlam, HyperBeam, Reflect, Rest}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<Set, 6> team9 = {
    Set{Species::Alakazam, {Psychic, Recover, SeismicToss, ThunderWave}},
    {Species::Chansey, {Reflect, SeismicToss, SoftBoiled, ThunderWave}},
    {Species::Exeggutor, {Explosion, HyperBeam, Psychic, SleepPowder}},
    {Species::Snorlax, {BodySlam, Earthquake, HyperBeam, SelfDestruct}},
    {Species::Starmie, {Blizzard, Recover, Thunderbolt, ThunderWave}},
    {Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}}};

std::array<std::array<Set, 6>, 10> teams{team0, team1, team2, team3, team4,
                                         team5, team6, team7, team8, team9};
}; // namespace SampleTeams

int rollout_sample_teams_and_save_log(int argc, char **argv) {
  constexpr size_t log_size{64};
  if (argc != 4) {
    std::cout
        << "Usage: provide two sample team indices [0 - 9] and a u64 seed."
        << std::endl;
    return 1;
  }
  int p1 = std::atoi(argv[1]);
  int p2 = std::atoi(argv[2]);
  uint64_t seed = std::atoi(argv[3]);
  prng device{seed};
  Battle<log_size, false, false> battle{SampleTeams::teams[p1],
                                        SampleTeams::teams[p2], seed};

  DebugLog<log_size> debug_log{};
  debug_log.rollout_battle(battle, device);
  debug_log.save_data_to_path("./" + std::to_string(seed));

  return 0;
}

int main(int argc, char **argv) {
  return rollout_sample_teams_and_save_log(argc, argv);
}
