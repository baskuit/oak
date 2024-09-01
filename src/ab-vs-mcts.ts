import 'source-map-support/register';

import { Generations } from '@pkmn/data';
import { Dex } from '@pkmn/dex';
import { Battle, Choice, Log, Lookup, Result } from '@pkmn/engine';
import { Team } from '@pkmn/sets';

import { spawn } from 'child_process';
import { decode } from 'punycode';

import { readFileSync } from 'fs';

// Globals
const TEAMS = readFileSync('rby.tsv', 'utf8').split('\n');
const N_TEAMS = TEAMS.length;

// Sorry lol
function result_to_byte(result: Result): number {
  let b = 0;
  switch (result.type) {
    case undefined:
      break;
    case 'win':
      b += 1;
      break;
    case 'lose':
      b += 2;
      break;
    case 'tie':
      b += 3;
      break;
    case 'error':
      b += 4;
      break;
  }
  switch (result.p1) {
    case 'pass':
      break;
    case 'move':
      b += 16;
      break;
    case 'switch':
      b += 32;
      break;
  }
  switch (result.p2) {
    case 'pass':
      break;
    case 'move':
      b += 64;
      break;
    case 'switch':
      b += 128;
      break;
  }
  return b;
}

function battle_bytes_string(battle: Battle, result: Result): string {
  const data: DataView = battle.data;
  let str: string = "";
  for (let i = 0; i < 384; ++i) {
    str += (data.getUint8(i).toString() + " ");
  }
  str += result_to_byte(result) + " ";
  return str;
}

function read_policies(res: string): number[][] {
  const get_policy = (line: string) => {
    let p: number[] = [];
    const words: string[] = line.split(' ');
    for (const word of words) {
      if (word === "") {
        continue
      }
      const x: number = Number(word);
      console.assert(0 <= x && x <= 1);
      p.push(x);
    }
    return p;
  };
  const lines: string[] = res.split('\n').filter(x => (x !== ""));
  return lines.map(line => get_policy(line));
}

class Random {
  seed: number;

  constructor(seed = 0x27d4eb2d) {
    this.seed = seed;
  }

  next(max: number) {
    let z = (this.seed += 0x6d2b79f5 | 0);
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z = z ^ (z + Math.imul(z ^ (z >>> 7), z | 61));
    z = (z ^ (z >>> 14)) >>> 0;
    const n = z / 0x100000000;
    return Math.floor(n * max);
  }
}

async function compare_teams_via_main(p1: number, p2: number) {

  const P1 = Team.unpack(TEAMS[0], Dex)!.team;
  const P2 = Team.unpack(TEAMS[1], Dex)!.team;

  const gens = new Generations(Dex);
  const gen = gens.get(1);
  const options = {
    p1: { name: 'Player A', team: P1 },
    p2: { name: 'Player B', team: P2 },
    seed: [1, 2, 3, 4],
    showdown: true,
    log: true,
  };
  const log = new Log(gen, Lookup.get(gen), options);

  let battle = Battle.create(gen, options);
  let result = battle.update(Choice.pass(), Choice.pass());

  const executablePath = './build/main';
  let child = spawn(executablePath);

  let ended = false;

  child.stdout.on('data', (data) => {
    const res: string = `${data}`;

    if (res[0] === '!') {
      console.log(res);
    } else {
      console.log("policies rec'd")
      const policies: number[][] = read_policies(res);
      const actions = policies.map(policy => {
        let p = 1;
        for (let i = 0; i < policy.length; i++) {
          p -= policy[i];
          if (p <= 0) {
            return i;
          }
        }
        return 0;
      }
      );

      console.log("policies: ", policies[0], policies[3]);
      console.log("actions: ", actions[0], actions[3]);

      const c1: Choice = battle.choices('p1', result)[actions[0]];
      const c2: Choice = battle.choices('p2', result)[actions[3]];

      result = battle.update(c1, c2);

      if (result.type) {
        console.log("battle ended");
        child.kill();
        ended = true;
        return;
      }

      const input: string = battle_bytes_string(battle, result);

      child.stdin.write(input);
    }

  });

  const initial_input = battle_bytes_string(battle, result);
  console.log(initial_input);
  child.stdin.write(initial_input);

  while (!ended) {
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

(async function () {
  await compare_teams_via_main(0, 1);
})();
