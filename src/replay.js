
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

            // Handle data chunks
            response.on('data', (chunk) => {
                data += chunk;
            });

            // Handle end of response
            response.on('end', () => {
                resolve(data);
            });
        }).on('error', (error) => {
            reject(`Error fetching URL: ${error.message}`);
        });
    });
}

function loadReplay (url) {
    (async () => {
        try {
            const log = await getUrlContent(url);
            const lines = log.split("\n");
            let battle = new client.Battle(new data.Generations(dex.Dex));
            for (let i = 0; i < lines.length; i = i + 1) {
                const line = lines[i];
                battle.add(line);
            }

            const gens = new data.Generations(dex.Dex);
            const gen = gens.get(1);
            const options =  {
                seed : [0,0,0,0,0,0],
                showdown : true,
                p1 : {name : "p1", team : battle.p1.team},
                p2 : {name : "p2", team : battle.p2.team},           
            };

            engine.Battle.create(gen, options);

        } catch (error) {
            console.error(error);
        }
    })();    
}

loadReplay("https://replay.pokemonshowdown.com/smogtours-gen1ou-742822");