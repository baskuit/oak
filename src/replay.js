
const client = require("@pkmn/client");
const data = require("@pkmn/data");
const de = require("@pkmn/dex");

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
            let battle = new client.Battle(new data.Generations(de.Dex));
            for (let i = 0; i < lines.length; i = i + 1) {
                const line = lines[i];
                battle.add(line);
                // console.log(line);
            }
            
            
        } catch (error) {
            console.error(error);
        }
    })();
    
}

loadReplay("https://replay.pokemonshowdown.com/smogtours-gen1ou-742822");