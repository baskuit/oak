#include <fstream>
#include <iostream>
#include <vector>

#include "../include/battle.hh"

std::array<std::string, 166> move_names{
    "None",         "Pound",        "KarateChop",   "DoubleSlap",   "CometPunch",   "MegaPunch",   "PayDay",
    "FirePunch",    "IcePunch",     "ThunderPunch", "Scratch",      "ViseGrip",     "Guillotine",  "RazorWind",
    "SwordsDance",  "Cut",          "Gust",         "WingAttack",   "Whirlwind",    "Fly",         "Bind",
    "Slam",         "VineWhip",     "Stomp",        "DoubleKick",   "MegaKick",     "JumpKick",    "RollingKick",
    "SandAttack",   "Headbutt",     "HornAttack",   "FuryAttack",   "HornDrill",    "Tackle",      "BodySlam",
    "Wrap",         "TakeDown",     "Thrash",       "DoubleEdge",   "TailWhip",     "PoisonSting", "Twineedle",
    "PinMissile",   "Leer",         "Bite",         "Growl",        "Roar",         "Sing",        "Supersonic",
    "SonicBoom",    "Disable",      "Acid",         "Ember",        "Flamethrower", "Mist",        "WaterGun",
    "HydroPump",    "Surf",         "IceBeam",      "Blizzard",     "Psybeam",      "BubbleBeam",  "AuroraBeam",
    "HyperBeam",    "Peck",         "DrillPeck",    "Submission",   "LowKick",      "Counter",     "SeismicToss",
    "Strength",     "Absorb",       "MegaDrain",    "LeechSeed",    "Growth",       "RazorLeaf",   "SolarBeam",
    "PoisonPowder", "StunSpore",    "SleepPowder",  "PetalDance",   "StringShot",   "DragonRage",  "FireSpin",
    "ThunderShock", "Thunderbolt",  "ThunderWave",  "Thunder",      "RockThrow",    "Earthquake",  "Fissure",
    "Dig",          "Toxic",        "Confusion",    "Psychic",      "Hypnosis",     "Meditate",    "Agility",
    "QuickAttack",  "Rage",         "Teleport",     "NightShade",   "Mimic",        "Screech",     "DoubleTeam",
    "Recover",      "Harden",       "Minimize",     "Smokescreen",  "ConfuseRay",   "Withdraw",    "DefenseCurl",
    "Barrier",      "LightScreen",  "Haze",         "Reflect",      "FocusEnergy",  "Bide",        "Metronome",
    "MirrorMove",   "SelfDestruct", "EggBomb",      "Lick",         "Smog",         "Sludge",      "BoneClub",
    "FireBlast",    "Waterfall",    "Clamp",        "Swift",        "SkullBash",    "SpikeCannon", "Constrict",
    "Amnesia",      "Kinesis",      "SoftBoiled",   "HighJumpKick", "Glare",        "DreamEater",  "PoisonGas",
    "Barrage",      "LeechLife",    "LovelyKiss",   "SkyAttack",    "Transform",    "Bubble",      "DizzyPunch",
    "Spore",        "Flash",        "Psywave",      "Splash",       "AcidArmor",    "Crabhammer",  "Explosion",
    "FurySwipes",   "Bonemerang",   "Rest",         "RockSlide",    "HyperFang",    "Sharpen",     "Conversion",
    "TriAttack",    "SuperFang",    "Slash",        "Substitute",   "Struggle"};

std::array<std::string, 152> species_names{
    "None",       "Bulbasaur",  "Ivysaur",   "Venusaur",   "Charmander", "Charmeleon", "Charizard",  "Squirtle",
    "Wartortle",  "Blastoise",  "Caterpie",  "Metapod",    "Butterfree", "Weedle",     "Kakuna",     "Beedrill",
    "Pidgey",     "Pidgeotto",  "Pidgeot",   "Rattata",    "Raticate",   "Spearow",    "Fearow",     "Ekans",
    "Arbok",      "Pikachu",    "Raichu",    "Sandshrew",  "Sandslash",  "NidoranF",   "Nidorina",   "Nidoqueen",
    "NidoranM",   "Nidorino",   "Nidoking",  "Clefairy",   "Clefable",   "Vulpix",     "Ninetales",  "Jigglypuff",
    "Wigglytuff", "Zubat",      "Golbat",    "Oddish",     "Gloom",      "Vileplume",  "Paras",      "Parasect",
    "Venonat",    "Venomoth",   "Diglett",   "Dugtrio",    "Meowth",     "Persian",    "Psyduck",    "Golduck",
    "Mankey",     "Primeape",   "Growlithe", "Arcanine",   "Poliwag",    "Poliwhirl",  "Poliwrath",  "Abra",
    "Kadabra",    "Alakazam",   "Machop",    "Machoke",    "Machamp",    "Bellsprout", "Weepinbell", "Victreebel",
    "Tentacool",  "Tentacruel", "Geodude",   "Graveler",   "Golem",      "Ponyta",     "Rapidash",   "Slowpoke",
    "Slowbro",    "Magnemite",  "Magneton",  "Farfetchd",  "Doduo",      "Dodrio",     "Seel",       "Dewgong",
    "Grimer",     "Muk",        "Shellder",  "Cloyster",   "Gastly",     "Haunter",    "Gengar",     "Onix",
    "Drowzee",    "Hypno",      "Krabby",    "Kingler",    "Voltorb",    "Electrode",  "Exeggcute",  "Exeggutor",
    "Cubone",     "Marowak",    "Hitmonlee", "Hitmonchan", "Lickitung",  "Koffing",    "Weezing",    "Rhyhorn",
    "Rhydon",     "Chansey",    "Tangela",   "Kangaskhan", "Horsea",     "Seadra",     "Goldeen",    "Seaking",
    "Staryu",     "Starmie",    "MrMime",    "Scyther",    "Jynx",       "Electabuzz", "Magmar",     "Pinsir",
    "Tauros",     "Magikarp",   "Gyarados",  "Lapras",     "Ditto",      "Eevee",      "Vaporeon",   "Jolteon",
    "Flareon",    "Porygon",    "Omanyte",   "Omastar",    "Kabuto",     "Kabutops",   "Aerodactyl", "Snorlax",
    "Articuno",   "Zapdos",     "Moltres",   "Dratini",    "Dragonair",  "Dragonite",  "Mewtwo",     "Mew"};

struct Frame {
    std::array<uint8_t, SIZE_BATTLE_WITH_PRNG> battle{};
    pkmn_result result;
    pkmn_choice row_action, col_action;
    std::vector<float> row_policy{}, col_policy{};
    std::vector<float> encoding{};
    float row_value;
};

struct Trajectory {
    float terminal_value;
    std::vector<Frame> frames{};
};

Trajectory get_trajectory(const char* data, const int size) {
    static size_t SIZE_SELF_PLAY_FRAME = SIZE_BATTLE_WITH_PRNG + 3;

    char final_result;

    Trajectory trajectory{};
    uint8_t next_battle[SIZE_BATTLE_WITH_PRNG];
    pkmn_result next_result{};

    trajectory.frames.emplace_back();
    memcpy(next_battle, data + 4, SIZE_BATTLE_WITH_PRNG);

    int frame_index = 0;

    // header info + battle
    int index{4 + SIZE_BATTLE_WITH_PRNG};
    while (index < size) {
        Frame& frame = trajectory.frames[frame_index];
        memcpy(frame.battle.data(), next_battle, SIZE_BATTLE_WITH_PRNG);

        memcpy(next_battle, data + index, SIZE_BATTLE_WITH_PRNG);
        index += SIZE_BATTLE_WITH_PRNG;

        // result
        final_result = *(data + index);
        frame.result = next_result;
        next_result = final_result;
        // std::cout << "result: " << (int)result << std::endl;
        index += 1;

        // row action
        frame.row_action = *(data + index);
        // std::cout << "row_action: " << (int)frame.row_action << std::endl;
        index += 1;

        // col action
        frame.col_action = *(data + index);
        // std::cout << "col_action: " << (int)frame.col_action << std::endl;
        index += 1;

        const int rows = *(data + index);
        // std::cout << "rows: " << (int)rows << std::endl;
        frame.row_policy.resize(rows);
        index += 1;

        const int cols = *(data + index);
        // std::cout << "cols: " << (int)cols << std::endl;
        frame.col_policy.resize(cols);
        index += 1;

        const float* float_ptr = reinterpret_cast<const float*>(data + index);
        frame.row_value = *float_ptr;
        ++float_ptr;
        for (int row_idx{}; row_idx < frame.row_policy.size(); ++row_idx) {
            frame.row_policy[row_idx] = *float_ptr;
            ++float_ptr;
        }
        for (int col_idx{}; col_idx < frame.col_policy.size(); ++col_idx) {
            frame.col_policy[col_idx] = *float_ptr;
            ++float_ptr;
        }

        // for both players: value + row_policy + col_policy
        const int total_floats = 2 * (1 + frame.row_policy.size() + frame.col_policy.size());
        // skip bytes for floats plus both n_matrix bytes (zero'd)
        const int total_bytes_skipped = 4 * total_floats + 2;
        index += total_bytes_skipped;

        ++frame_index;
        trajectory.frames.emplace_back();
    }

    assert(index == size);

    switch (pkmn_result_kind(final_result)) {
        case PKMN_RESULT_WIN: {
            trajectory.terminal_value = 1.0;
        }
        case PKMN_RESULT_LOSE: {
            trajectory.terminal_value = 0.0;
        }
        case PKMN_RESULT_TIE: {
            trajectory.terminal_value = 0.5;
        }
        case PKMN_RESULT_ERROR: {
            std::exception();
        }
    }
    // get rid of empty frame
    trajectory.frames.pop_back();

    return trajectory;
}

void encode_volatiles(const uint8_t* data, std::vector<float>& encoding) {
    for (int byte{}; byte < 2; ++byte) {
        for (int bit{}; bit < 8; ++bit) {
            encoding.push_back(static_cast<float>(data[byte] & (1 << bit)));
        }
    }

    encoding.push_back(static_cast<float>(data[2] & 1));             // reflect
    encoding.push_back(static_cast<float>(data[2] & 2));             // transform
    encoding.push_back(static_cast<float>(data[2] & (4 + 8 + 16)));  // confusion dur
    encoding.push_back(static_cast<float>(data[2] >> 5));            // attacks dur
    // TODO state still not clear to me, but this probably works
    const auto* state_ptr = reinterpret_cast<const uint16_t*>(data + 3);
    encoding.push_back(static_cast<float>(*state_ptr));
    encoding.push_back(static_cast<float>(data[5]));       // sub hp
    encoding.push_back(static_cast<float>(data[6] & 15));  // transform id
    encoding.push_back(static_cast<float>(data[6] >> 4));  // disable dur
    encoding.push_back(static_cast<float>(data[7] & 7));   // disabled move
    encoding.push_back(static_cast<float>(data[7] >> 3));  // toxic turns
}

void encode_active_pokemon(const uint8_t* data, std::vector<float>& encoding) {
    const auto* data_16 = reinterpret_cast<const uint16_t*>(data);
    encoding.push_back(static_cast<float>(data_16[0]));  // hp
    encoding.push_back(static_cast<float>(data_16[1]));  // atk
    encoding.push_back(static_cast<float>(data_16[2]));  // def
    encoding.push_back(static_cast<float>(data_16[3]));  // spe
    encoding.push_back(static_cast<float>(data_16[4]));  // spc
    encoding.push_back(static_cast<float>(data[10]));    // species
    uint8_t type = data[11];
    encoding.push_back(static_cast<float>(type & 15));  // type 1
    encoding.push_back(static_cast<float>(type >> 4));  // type 2
    uint8_t boosts_atk_def = data[12];
    encoding.push_back(static_cast<float>(boosts_atk_def & 15));  // atk boost
    encoding.push_back(static_cast<float>(boosts_atk_def >> 4));  // def boost
    uint8_t boosts_spe_spc = data[13];
    encoding.push_back(static_cast<float>(boosts_spe_spc & 15));  // spe boost
    encoding.push_back(static_cast<float>(boosts_spe_spc >> 4));  // spc boost
    uint8_t boosts_acc_eva = data[14];
    encoding.push_back(static_cast<float>(boosts_acc_eva & 15));  // acc boost
    encoding.push_back(static_cast<float>(boosts_acc_eva >> 4));  // eva boost
    // data[15] is zero padding
    encode_volatiles(data + 16, encoding);

    for (int i{}; i < 8; ++i) {
        encoding.push_back(static_cast<float>(data[24 + i]));
    }
}

void encode_bench_pokemon(const uint8_t* data, std::vector<float>& encoding) {
    const auto* data_16 = reinterpret_cast<const uint16_t*>(data);
    encoding.push_back(static_cast<float>(data_16[0]));  // hp
    encoding.push_back(static_cast<float>(data_16[1]));  // atk
    encoding.push_back(static_cast<float>(data_16[2]));  // def
    encoding.push_back(static_cast<float>(data_16[3]));  // spe
    encoding.push_back(static_cast<float>(data_16[4]));  // spc
    // 4 moves: id + pp
    for (int i{}; i < 8; ++i) {
        encoding.push_back(static_cast<float>(data[10 + i]));
    }
    encoding.push_back(static_cast<float>(data_16[9]));   // current hp
    const uint8_t status = data[20];                      // status
    encoding.push_back(static_cast<float>(status & 7));   // status duration
    encoding.push_back(static_cast<float>(status >> 3));  // status id

    encoding.push_back(static_cast<float>(data[21]));  // species
    const uint8_t type = data[22];
    encoding.push_back(static_cast<float>(type & 15));  // type 1
    encoding.push_back(static_cast<float>(type >> 4));  // type 2

    encoding.push_back(static_cast<float>(data[23]));  // level
}

void encode_side(const uint8_t* data, std::vector<float>& encoding) {
    // last used move is outside this call, in encode_battle
    const uint8_t* order = data + 176;

    const uint8_t* active_data = data + 144;
    encode_active_pokemon(active_data, encoding);

    for (int p{}; p < 6; ++p) {
        const uint8_t pokemon = order[p] - 1;
        assert(pokemon >= 0 and pokemon < 6);
        const uint8_t* pokemon_data = data + 24 * pokemon;
        encode_bench_pokemon(pokemon_data, encoding);
    }
}

void encode_battle(const uint8_t* data, std::vector<float>& encoding) {
    for (int side{}; side < 2; ++side) {
        const uint8_t* side_data = data + side * SIZE_SIDE;
        encode_side(side_data, encoding);
        encoding.push_back(static_cast<float>(data[372 + 2 * side]));  // last move
        encoding.push_back(static_cast<float>(data[373 + 2 * side]));  // counterable
    }
    encoding.push_back(static_cast<float>(data[368] + 256 * data[369]));  // turn
    encoding.push_back(static_cast<float>(data[370] + 256 * data[371]));  // last damage
}

std::string decode_action(const uint8_t* side_data, const pkmn_choice action) {
    const int data = action >> 2;
    // std::cout << (int)(action & 3) << ' ' << (int)(action >> 2) << std::endl;
    switch (action & 3) {
        case 0: {
            return "Pass";
        }
        case 1: {
            const uint8_t* active_move_data = side_data + 144 + 24;
            const uint8_t move_id = active_move_data[2 * (data - 1)];
            return move_names[move_id];
        }
        case 2: {
            // const uint8_t* order = side_data + 176;
            const uint8_t* switch_target = side_data + 24 * (data - 1);
            const uint8_t switch_species = switch_target[21];
            return species_names[switch_species];
        }
    }
    return "ERROR DECODING";
}

Trajectory open_file_and_get_trajectory(std::string file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return {};
    }

    // Get the size of the file
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file into a vector
    std::vector<char> fileData(fileSize);
    file.read(fileData.data(), fileSize);

    // Get a pointer to the file data
    char* data = fileData.data();

    // Output the number of bytes in the file
    std::cout << "Number of bytes in the file: " << fileSize << std::endl;

    // You can now use 'fileDataPtr' to access the bytes of the file

    // Don't forget to close the file
    file.close();

    const Trajectory trajectory = get_trajectory(data, fileSize);

    return trajectory;
}

int main() {
    std::string demo_path = "/home/user/oak/logs/12463111002853059008.log";
    auto trajectory = open_file_and_get_trajectory(demo_path);

    int frame_index{};
    for (Frame& frame : trajectory.frames) {
        std::cout << "Frame: " << frame_index << std::endl;
        encode_battle(frame.battle.data(), frame.encoding);

        Battle<0, 0, BattleObs, bool, float>::State state{frame.battle.data(), frame.battle.data() + SIZE_SIDE};
        state.result = frame.result;
        std::cout << "result: " << (int)frame.result << std::endl;
        state.get_actions();

        assert(frame.row_policy.size() == state.row_actions.size());
        assert(frame.col_policy.size() == state.col_actions.size());

        std::cout << "row actions:" << std::endl;
        for (const auto action : state.row_actions) {
            std::cout << decode_action(frame.battle.data(), action) << ", ";
        }
        std::cout << std::endl;

        std::cout << "col actions:" << std::endl;
        for (const auto action : state.col_actions) {
            std::cout << decode_action(frame.battle.data() + SIZE_SIDE, action) << ", ";
        }
        std::cout << std::endl;

        ++frame_index;
    }
    return 0;
}