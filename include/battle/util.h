#pragma once

#include <pkmn.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <data/data.h>

namespace Names {
constexpr std::array<std::string, 166> MOVE_STRING{
    "None",         "Pound",        "KarateChop",  "DoubleSlap",
    "CometPunch",   "MegaPunch",    "PayDay",      "FirePunch",
    "IcePunch",     "ThunderPunch", "Scratch",     "ViseGrip",
    "Guillotine",   "RazorWind",    "SwordsDance", "Cut",
    "Gust",         "WingAttack",   "Whirlwind",   "Fly",
    "Bind",         "Slam",         "VineWhip",    "Stomp",
    "DoubleKick",   "MegaKick",     "JumpKick",    "RollingKick",
    "SandAttack",   "Headbutt",     "HornAttack",  "FuryAttack",
    "HornDrill",    "Tackle",       "BodySlam",    "Wrap",
    "TakeDown",     "Thrash",       "DoubleEdge",  "TailWhip",
    "PoisonSting",  "Twineedle",    "PinMissile",  "Leer",
    "Bite",         "Growl",        "Roar",        "Sing",
    "Supersonic",   "SonicBoom",    "Disable",     "Acid",
    "Ember",        "Flamethrower", "Mist",        "WaterGun",
    "HydroPump",    "Surf",         "IceBeam",     "Blizzard",
    "Psybeam",      "BubbleBeam",   "AuroraBeam",  "HyperBeam",
    "Peck",         "DrillPeck",    "Submission",  "LowKick",
    "Counter",      "SeismicToss",  "Strength",    "Absorb",
    "MegaDrain",    "LeechSeed",    "Growth",      "RazorLeaf",
    "SolarBeam",    "PoisonPowder", "StunSpore",   "SleepPowder",
    "PetalDance",   "StringShot",   "DragonRage",  "FireSpin",
    "ThunderShock", "Thunderbolt",  "ThunderWave", "Thunder",
    "RockThrow",    "Earthquake",   "Fissure",     "Dig",
    "Toxic",        "Confusion",    "Psychic",     "Hypnosis",
    "Meditate",     "Agility",      "QuickAttack", "Rage",
    "Teleport",     "NightShade",   "Mimic",       "Screech",
    "DoubleTeam",   "Recover",      "Harden",      "Minimize",
    "Smokescreen",  "ConfuseRay",   "Withdraw",    "DefenseCurl",
    "Barrier",      "LightScreen",  "Haze",        "Reflect",
    "FocusEnergy",  "Bide",         "Metronome",   "MirrorMove",
    "SelfDestruct", "EggBomb",      "Lick",        "Smog",
    "Sludge",       "BoneClub",     "FireBlast",   "Waterfall",
    "Clamp",        "Swift",        "SkullBash",   "SpikeCannon",
    "Constrict",    "Amnesia",      "Kinesis",     "SoftBoiled",
    "HighJumpKick", "Glare",        "DreamEater",  "PoisonGas",
    "Barrage",      "LeechLife",    "LovelyKiss",  "SkyAttack",
    "Transform",    "Bubble",       "DizzyPunch",  "Spore",
    "Flash",        "Psywave",      "Splash",      "AcidArmor",
    "Crabhammer",   "Explosion",    "FurySwipes",  "Bonemerang",
    "Rest",         "RockSlide",    "HyperFang",   "Sharpen",
    "Conversion",   "TriAttack",    "SuperFang",   "Slash",
    "Substitute",   "Struggle"};

constexpr std::string move_string(const Data::Moves move) {
  return MOVE_STRING[static_cast<uint8_t>(move)];
}

constexpr std::array<std::string, 152> SPECIES_STRING{
    "None",       "Bulbasaur",  "Ivysaur",    "Venusaur",   "Charmander",
    "Charmeleon", "Charizard",  "Squirtle",   "Wartortle",  "Blastoise",
    "Caterpie",   "Metapod",    "Butterfree", "Weedle",     "Kakuna",
    "Beedrill",   "Pidgey",     "Pidgeotto",  "Pidgeot",    "Rattata",
    "Raticate",   "Spearow",    "Fearow",     "Ekans",      "Arbok",
    "Pikachu",    "Raichu",     "Sandshrew",  "Sandslash",  "NidoranF",
    "Nidorina",   "Nidoqueen",  "NidoranM",   "Nidorino",   "Nidoking",
    "Clefairy",   "Clefable",   "Vulpix",     "Ninetales",  "Jigglypuff",
    "Wigglytuff", "Zubat",      "Golbat",     "Oddish",     "Gloom",
    "Vileplume",  "Paras",      "Parasect",   "Venonat",    "Venomoth",
    "Diglett",    "Dugtrio",    "Meowth",     "Persian",    "Psyduck",
    "Golduck",    "Mankey",     "Primeape",   "Growlithe",  "Arcanine",
    "Poliwag",    "Poliwhirl",  "Poliwrath",  "Abra",       "Kadabra",
    "Alakazam",   "Machop",     "Machoke",    "Machamp",    "Bellsprout",
    "Weepinbell", "Victreebel", "Tentacool",  "Tentacruel", "Geodude",
    "Graveler",   "Golem",      "Ponyta",     "Rapidash",   "Slowpoke",
    "Slowbro",    "Magnemite",  "Magneton",   "Farfetchd",  "Doduo",
    "Dodrio",     "Seel",       "Dewgong",    "Grimer",     "Muk",
    "Shellder",   "Cloyster",   "Gastly",     "Haunter",    "Gengar",
    "Onix",       "Drowzee",    "Hypno",      "Krabby",     "Kingler",
    "Voltorb",    "Electrode",  "Exeggcute",  "Exeggutor",  "Cubone",
    "Marowak",    "Hitmonlee",  "Hitmonchan", "Lickitung",  "Koffing",
    "Weezing",    "Rhyhorn",    "Rhydon",     "Chansey",    "Tangela",
    "Kangaskhan", "Horsea",     "Seadra",     "Goldeen",    "Seaking",
    "Staryu",     "Starmie",    "MrMime",     "Scyther",    "Jynx",
    "Electabuzz", "Magmar",     "Pinsir",     "Tauros",     "Magikarp",
    "Gyarados",   "Lapras",     "Ditto",      "Eevee",      "Vaporeon",
    "Jolteon",    "Flareon",    "Porygon",    "Omanyte",    "Omastar",
    "Kabuto",     "Kabutops",   "Aerodactyl", "Snorlax",    "Articuno",
    "Zapdos",     "Moltres",    "Dratini",    "Dragonair",  "Dragonite",
    "Mewtwo",     "Mew"};

constexpr std::string species_string(const Data::Species species) {
  return SPECIES_STRING[static_cast<uint8_t>(species)];
}

}; // namespace Names

void print_moves(const uint8_t *pokemon) {
  for (int m = 10; m < 18; m += 2) {
    std::cout << Names::MOVE_STRING[pokemon[m]] << std::endl;
  }
}

void print_species(const uint8_t *pokemon) {
  std::cout << Names::SPECIES_STRING[pokemon[21]] << std::endl;
}

constexpr const uint8_t *get_pokemon_from_slot(const uint8_t *side,
                                               int slot = 1) {
  const auto index = side[175 + slot] - 1;
  return side + 24 * index;
}

constexpr const uint8_t *order_bits(const uint8_t *side) { return side + 176; }

constexpr std::string side_choice_string(const uint8_t *side,
                                         pkmn_choice choice) {
  const auto choice_type = choice & 3;
  const auto choice_data = choice >> 2;
  switch (choice_type) {
  case 0: {
    return "pass";
  }
  case 1: {
    return Names::MOVE_STRING[get_pokemon_from_slot(side,
                                                    1)[8 + 2 * choice_data]];
  }
  case 2: {
    return Names::SPECIES_STRING[get_pokemon_from_slot(side, choice_data)[21]];
  }
  default: {
    std::cout << "bad choice - data: " << (int)choice_data
              << " type: " << (int)choice_type << std::endl;
    return "";
  }
  }
}

constexpr std::string buffer_to_string(const uint8_t *const buf, int n) {
  std::string result;
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      result += ' ';
    }
    int value = static_cast<int>(buf[i]);
    char temp[4] = {};
    int len = 0;

    // Convert integer to string representation without to_string
    if (value == 0) {
      temp[len++] = '0';
    } else {
      int temp_val = value;
      while (temp_val > 0) {
        temp[len++] = '0' + (temp_val % 10);
        temp_val /= 10;
      }
      // Reverse the characters in place
      for (int j = 0; j < len / 2; ++j) {
        std::swap(temp[j], temp[len - j - 1]);
      }
    }
    result.append(temp, len);
  }
  return result;
}

// maybe pinyon this, its basically a chance node
template <typename Key, typename Value> struct LinearScanMap {
  std::vector<std::pair<Key, Value>> data;

  Value &operator[](const Key &key) {
    for (auto &pair : data) {
      if (pair.first == key) {
        return pair.second;
      }
    }
    data.emplace_back(key, Value{});
    return data.back().second;
  }

  const Value *at(const Key &key) const noexcept {
    for (const auto &pair : data) {
      if (pair.first == key) {
        return &pair.second;
      }
    }
    return nullptr;
  }

  constexpr auto size() const noexcept { return data.size(); }
};
