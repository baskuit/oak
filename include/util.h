#pragma once

#include <pkmn.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace Helpers {

enum class Species {
  None,
  Bulbasaur,
  Ivysaur,
  Venusaur,
  Charmander,
  Charmeleon,
  Charizard,
  Squirtle,
  Wartortle,
  Blastoise,
  Caterpie,
  Metapod,
  Butterfree,
  Weedle,
  Kakuna,
  Beedrill,
  Pidgey,
  Pidgeotto,
  Pidgeot,
  Rattata,
  Raticate,
  Spearow,
  Fearow,
  Ekans,
  Arbok,
  Pikachu,
  Raichu,
  Sandshrew,
  Sandslash,
  NidoranF,
  Nidorina,
  Nidoqueen,
  NidoranM,
  Nidorino,
  Nidoking,
  Clefairy,
  Clefable,
  Vulpix,
  Ninetales,
  Jigglypuff,
  Wigglytuff,
  Zubat,
  Golbat,
  Oddish,
  Gloom,
  Vileplume,
  Paras,
  Parasect,
  Venonat,
  Venomoth,
  Diglett,
  Dugtrio,
  Meowth,
  Persian,
  Psyduck,
  Golduck,
  Mankey,
  Primeape,
  Growlithe,
  Arcanine,
  Poliwag,
  Poliwhirl,
  Poliwrath,
  Abra,
  Kadabra,
  Alakazam,
  Machop,
  Machoke,
  Machamp,
  Bellsprout,
  Weepinbell,
  Victreebel,
  Tentacool,
  Tentacruel,
  Geodude,
  Graveler,
  Golem,
  Ponyta,
  Rapidash,
  Slowpoke,
  Slowbro,
  Magnemite,
  Magneton,
  Farfetchd,
  Doduo,
  Dodrio,
  Seel,
  Dewgong,
  Grimer,
  Muk,
  Shellder,
  Cloyster,
  Gastly,
  Haunter,
  Gengar,
  Onix,
  Drowzee,
  Hypno,
  Krabby,
  Kingler,
  Voltorb,
  Electrode,
  Exeggcute,
  Exeggutor,
  Cubone,
  Marowak,
  Hitmonlee,
  Hitmonchan,
  Lickitung,
  Koffing,
  Weezing,
  Rhyhorn,
  Rhydon,
  Chansey,
  Tangela,
  Kangaskhan,
  Horsea,
  Seadra,
  Goldeen,
  Seaking,
  Staryu,
  Starmie,
  MrMime,
  Scyther,
  Jynx,
  Electabuzz,
  Magmar,
  Pinsir,
  Tauros,
  Magikarp,
  Gyarados,
  Lapras,
  Ditto,
  Eevee,
  Vaporeon,
  Jolteon,
  Flareon,
  Porygon,
  Omanyte,
  Omastar,
  Kabuto,
  Kabutops,
  Aerodactyl,
  Snorlax,
  Articuno,
  Zapdos,
  Moltres,
  Dratini,
  Dragonair,
  Dragonite,
  Mewtwo,
  Mew
};

enum class Moves {
  None,
  Pound,
  KarateChop,
  DoubleSlap,
  CometPunch,
  MegaPunch,
  PayDay,
  FirePunch,
  IcePunch,
  ThunderPunch,
  Scratch,
  ViseGrip,
  Guillotine,
  RazorWind,
  SwordsDance,
  Cut,
  Gust,
  WingAttack,
  Whirlwind,
  Fly,
  Bind,
  Slam,
  VineWhip,
  Stomp,
  DoubleKick,
  MegaKick,
  JumpKick,
  RollingKick,
  SandAttack,
  Headbutt,
  HornAttack,
  FuryAttack,
  HornDrill,
  Tackle,
  BodySlam,
  Wrap,
  TakeDown,
  Thrash,
  DoubleEdge,
  TailWhip,
  PoisonSting,
  Twineedle,
  PinMissile,
  Leer,
  Bite,
  Growl,
  Roar,
  Sing,
  Supersonic,
  SonicBoom,
  Disable,
  Acid,
  Ember,
  Flamethrower,
  Mist,
  WaterGun,
  HydroPump,
  Surf,
  IceBeam,
  Blizzard,
  Psybeam,
  BubbleBeam,
  AuroraBeam,
  HyperBeam,
  Peck,
  DrillPeck,
  Submission,
  LowKick,
  Counter,
  SeismicToss,
  Strength,
  Absorb,
  MegaDrain,
  LeechSeed,
  Growth,
  RazorLeaf,
  SolarBeam,
  PoisonPowder,
  StunSpore,
  SleepPowder,
  PetalDance,
  StringShot,
  DragonRage,
  FireSpin,
  ThunderShock,
  Thunderbolt,
  ThunderWave,
  Thunder,
  RockThrow,
  Earthquake,
  Fissure,
  Dig,
  Toxic,
  Confusion,
  Psychic,
  Hypnosis,
  Meditate,
  Agility,
  QuickAttack,
  Rage,
  Teleport,
  NightShade,
  Mimic,
  Screech,
  DoubleTeam,
  Recover,
  Harden,
  Minimize,
  Smokescreen,
  ConfuseRay,
  Withdraw,
  DefenseCurl,
  Barrier,
  LightScreen,
  Haze,
  Reflect,
  FocusEnergy,
  Bide,
  Metronome,
  MirrorMove,
  SelfDestruct,
  EggBomb,
  Lick,
  Smog,
  Sludge,
  BoneClub,
  FireBlast,
  Waterfall,
  Clamp,
  Swift,
  SkullBash,
  SpikeCannon,
  Constrict,
  Amnesia,
  Kinesis,
  SoftBoiled,
  HighJumpKick,
  Glare,
  DreamEater,
  PoisonGas,
  Barrage,
  LeechLife,
  LovelyKiss,
  SkyAttack,
  Transform,
  Bubble,
  DizzyPunch,
  Spore,
  Flash,
  Psywave,
  Splash,
  AcidArmor,
  Crabhammer,
  Explosion,
  FurySwipes,
  Bonemerang,
  Rest,
  RockSlide,
  HyperFang,
  Sharpen,
  Conversion,
  TriAttack,
  SuperFang,
  Slash,
  Substitute,
  Struggle
};

struct Pokemon {
  Species species{};
  std::array<Moves, 4> moves{};

  bool operator==(const Pokemon &other) const noexcept = default;
};

using Side = std::array<Pokemon, 6>;
using Battle = std::array<Side, 2>;

bool RandbatObservationMatches(const Battle &seen, const Battle &omni) {

  const auto pokemon_match_almost = [](const Pokemon &a, const Pokemon &b) {
    if (a.species != b.species) {
      return false;
    }
    // todo optimize?
    for (int i = 0; i < 4; ++i) {
      if (a.moves[i] == Moves::None) {
        continue;
      }
      bool seen = false;
      for (int j = 0; j < 4; ++j) {
        seen = seen || (a.moves[i] == b.moves[j]);
      }
      if (!seen) {
        return false;
      }
    }
    return true;
  };

  const auto sides_match_almost = [](const Side &a, const Side &b) {
    for (const auto &pokemon : a) {
      if (pokemon.species == Species::None) {
        continue;
      }
      return false;
      // for (int i)
      // if (!pokemon_match_almost)
    }
    return true;
  };

  bool observer_can_be_p1 = true;
  bool observer_can_be_p2 = true;
  for (int side = 0; side < 2; ++side) {
    for (int pokemon = 0; pokemon < 6; ++pokemon) {
    }
  }

  return seen == omni;
}

struct prng {
  uint64_t _state;

  void next() {}
};

Battle generate(prng device) { return {}; }

bool test_generate(const uint64_t seed, const Battle &observed_battle) {
  return RandbatObservationMatches(observed_battle, generate(prng{seed}));
}

void init_battle(pkmn_gen1_battle *battle, const Battle &b) { return; }
}; // namespace Helpers

// pub const Pokemon = struct {
//     /// The Pokémon's species.
//     species: Species,
//     /// The Pokémon's moves (assumed to all have the max possible PP).
//     moves: []const Move,
//     /// The Pokémon's current HP (defaults to 100% if not specified).
//     hp: ?u16 = null,
//     /// The Pokémon's current status.
//     status: u8 = 0,
//     /// The Pokémon's level.
//     level: u8 = 100,
//     /// The Pokémon's DVs.
//     dvs: DVs = .{},
//     /// The Pokémon's stat experience.
//     stats: Stats(u16) = .{ .hp = EXP, .atk = EXP, .def = EXP, .spe = EXP,
//     .spc = EXP },

//     /// Initializes a Generation I Pokémon based on the information in `p`.
//     pub fn init(p: Pokemon) data.Pokemon {
//         var pokemon = data.Pokemon{};
//         pokemon.species = p.species;
//         const species = Species.get(p.species);
//         inline for (@field(@typeInfo(@TypeOf(pokemon.stats)),
//         @tagName(Struct)).fields) |field| {
//             const hp = comptime std.mem.eql(u8, field.name, "hp");
//             const spc =
//                 comptime std.mem.eql(u8, field.name, "spa") or
//                 std.mem.eql(u8, field.name, "spd");
//             @field(pokemon.stats, field.name) = Stats(u16).calc(
//                 field.name,
//                 @field(species.stats, field.name),
//                 if (hp) p.dvs.hp() else if (spc) p.dvs.spc else @field(p.dvs,
//                 field.name),
//                 @field(p.stats, field.name),
//                 p.level,
//             );
//         }
//         assert(p.moves.len > 0 and p.moves.len <= 4);
//         for (p.moves, 0..) |m, j| {
//             pokemon.moves[j].id = m;
//             // NB: PP can be at most 61 legally (though can overflow to 63)
//             pokemon.moves[j].pp = @intCast(@min(Move.pp(m) / 5 * 8, 61));
//         }
//         if (p.hp) |hp| {
//             pokemon.hp = hp;
//         } else {
//             pokemon.hp = pokemon.stats.hp;
//         }
//         pokemon.status = p.status;
//         pokemon.types = species.types;
//         pokemon.level = p.level;
//         return pokemon;
//     }

namespace Names {
static constexpr std::string move_name[]{
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

static constexpr std::string species_name[]{
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
}; // namespace Names

void print_moves(const uint8_t *pokemon) {
  for (int m = 10; m < 18; m += 2) {
    std::cout << Names::move_name[pokemon[m]] << std::endl;
  }
}

void print_species(const uint8_t *pokemon) {
  std::cout << Names::species_name[pokemon[21]] << std::endl;
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
    return Names::move_name[get_pokemon_from_slot(side,
                                                  1)[8 + 2 * choice_data]];
  }
  case 2: {
    return Names::species_name[get_pokemon_from_slot(side, choice_data)[21]];
  }
  default: {
    std::cout << "bad choice - data: " << (int)choice_data
              << " type: " << (int)choice_type << std::endl;
    return "";
  }
  }
}

constexpr std::string buffer_to_string(const uint8_t *const buf, int n) {
  std::stringstream stream{};
  for (int i = 0; i < n - 1; ++i) {
    stream << std::to_string(static_cast<int>(buf[i])) << ' ';
  }
  stream << std::to_string(static_cast<int>(buf[n - 1]));
  return stream.str();
}