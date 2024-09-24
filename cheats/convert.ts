const set_json = require("../extern/pokemon-showdown/data/random-battles/gen1/data.json");

let all_moves: Set<string> = new Set();

for (let species in set_json) {
    const data = set_json[species];
    const level: Number = data.level || 100;
    const moves: string[] = data.moves || [];
    const exclusiveMoves: string[] = data.exclusiveMoves || [];
    const essentialMoves: string[] = data.essentialMoves || [];
    const comboMoves: string[] = data.comboMoves || [];

    for (const thing of Object.values(data)) {
        if (Number.isInteger(thing)) {
            continue;
        }
        for (const move of thing as string[]) {
            all_moves.add(move);
        }
    }
}

// only needed to produce the map above
// console.log(all_moves
const all_moves_precomputed = ["bodyslam",
    "razorleaf",
    "sleeppowder",
    "swordsdance",
    "hyperbeam",
    "counter",
    "seismictoss",
    "slash",
    "fireblast",
    "submission",
    "earthquake",
    "blizzard",
    "hydropump",
    "surf",
    "rest",
    "psychic",
    "stunspore",
    "doubleedge",
    "megadrain",
    "substitute",
    "twineedle",
    "agility",
    "quickattack",
    "skyattack",
    "mirrormove",
    "sandattack",
    "reflect",
    "superfang",
    "thunderbolt",
    "drillpeck",
    "leer",
    "mimic",
    "glare",
    "rockslide",
    "thunderwave",
    "thunder",
    "doublekick",
    "bubblebeam",
    "sing",
    "confuseray",
    "flamethrower",
    "wingattack",
    "spore",
    "amnesia",
    "lowkick",
    "megakick",
    "hypnosis",
    "recover",
    "barrier",
    "explosion",
    "stomp",
    "sludge",
    "nightshade",
    "crabhammer",
    "takedown",
    "highjumpkick",
    "meditate",
    "rollingkick",
    "icebeam",
    "softboiled",
    "growth",
    "smokescreen",
    "lovelykiss",
    "transform",
    "tailwhip",
    "acidarmor",
    "pinmissile",
    "triattack",
    "selfdestruct"];
const all_moves_fixed = [
    "Moves::BodySlam",
    "Moves::RazorLeaf",
    "Moves::SleepPowder",
    "Moves::SwordDance",
    "Moves::HyperBeam",
    "Moves::Counter",
    "Moves::SeismicToss",
    "Moves::Slash",
    "Moves::FireBlast",
    "Moves::Submission",
    "Moves::Earthquake",
    "Moves::Blizzard",
    "Moves::HydroPump",
    "Moves::Surf",
    "Moves::Rest",
    "Moves::Psychic",
    "Moves::StunSpore",
    "Moves::DoubleEdge",
    "Moves::MegaDrain",
    "Moves::Substitute",
    "Moves::TwinNeedle",
    "Moves::Agility",
    "Moves::QuickAttack",
    "Moves::SkyAttack",
    "Moves::MirrorMove",
    "Moves::SandAttack",
    "Moves::Reflect",
    "Moves::SuperFang",
    "Moves::ThunderBolt",
    "Moves::DrillPeck",
    "Moves::Leer",
    "Moves::Mimic",
    "Moves::Glare",
    "Moves::RockSlide",
    "Moves::ThunderWave",
    "Moves::Thunder",
    "Moves::DoubleKick",
    "Moves::BubbleBeam",
    "Moves::Sing",
    "Moves::ConfuseRay",
    "Moves::FlameThrower",
    "Moves::WingAttack",
    "Moves::Spore",
    "Moves::Amnesia",
    "Moves::LowKick",
    "Moves::MegaKick",
    "Moves::Hypnosis",
    "Moves::Recover",
    "Moves::Barrier",
    "Moves::Explosion",
    "Moves::Stomp",
    "Moves::Sludge",
    "Moves::NightShade",
    "Moves::CrabHammer",
    "Moves::TakeDown",
    "Moves::HighJumpKick",
    "Moves::Meditate",
    "Moves::RollingKick",
    "Moves::IceBeam",
    "Moves::SoftBoiled",
    "Moves::Growth",
    "Moves::SmokeScreen",
    "Moves::LovelyKiss",
    "Moves::Transform",
    "Moves::TailWhip",
    "Moves::AcidArmor",
    "Moves::PinMissile",
    "Moves::TriAttack",
    "Moves::SelfDestruct"
];

function fixName(move: string): string {
    if (move === undefined) {
        return "Moves::None";
    }
    
    const index = all_moves_precomputed.indexOf(move);

    if (index === -1) {
        console.error("bad move string");
    }

    return all_moves_fixed[index];
}