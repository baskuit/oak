
const client = require("@pkmn/client");
const data = require("@pkmn/data");
const dex = require("@pkmn/dex");
const engine = require("@pkmn/engine")

const https = require('https');

function getUrlContent(url) {
    if (!url.endsWith('.log')) {
        url += '.log';
    }
    return new Promise((resolve, reject) => {
        https.get(url, (response) => {
            let data = '';

            response.on('data', (chunk) => {
                data += chunk;
            });

            response.on('end', () => {
                resolve(data);
            });
        }).on('error', (error) => {
            reject(`Error fetching URL: ${error.message}`);
        });
    });
}

function printEnginePokemon (p) {
    console.log(p.species, p.hp, p.status);
    for (const m of p.moves) {
        console.log(m);
    }
}

function printEngineBattle (battle) {
    for (const side of battle.sides) {
        console.log("\nSIDE:");
        for (const p of side.pokemon) {
            printEnginePokemon(p);
        }
    }
}

function printClientBattle (battle) {
    for (const side of [battle.p1, battle.p2]) {
        console.log('\n', side.name);
        for (const p of side.team) {
            console.log(p.name, p.hp, p.status, p);
        }
    }
}

function loadReplay(url, turn = 1000) {
    (async () => {
        try {
            const log = await getUrlContent(url);
            const lines = log.split("\n");
            let finished_battle = new client.Battle(new data.Generations(dex.Dex));
            let position = new client.Battle(new data.Generations(dex.Dex));

            for (let i = 0; i < lines.length; i = i + 1) {
                const line = lines[i];
                finished_battle.add(line);

                if (position.turn < turn) {
                    position.add(line);
                }
            }

            printClientBattle(position);
            printClientBattle(finished_battle);

            const gens = new data.Generations(dex.Dex);
            const gen = gens.get(1);
            const options = {
                seed: [0, 0, 0, 0, 0, 0],
                showdown: true,
                p1: { name: "p1", team: finished_battle.p1.team },
                p2: { name: "p2", team: finished_battle.p2.team },
            };

            // has mons and moves but no pp or hp
            var engine_battle = engine.Battle.create(gen, options);

            printEngineBattle(engine_battle);

        } catch (error) {
            console.error(error);
        }
    })();
}

loadReplay("https://replay.pokemonshowdown.com/smogtours-gen1ou-742822", 4);