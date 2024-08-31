import 'source-map-support/register';

import {Generations} from '@pkmn/data';
import {Dex} from '@pkmn/dex';
import {Battle, Choice, Log, Lookup, Result} from '@pkmn/engine';
import {Team} from '@pkmn/sets';

import { spawn } from 'child_process';
import { decode } from 'punycode';


function battle_bytes_string(battle : Battle, result : Result | null) : string 
{
  const data : DataView = battle.data;
  let str :string = "";
  for (let i = 0; i < 384; ++i) {
    str += (data.getUint8(i).toString() + " ");
  }
  // str += result.data.getUint8(0);
  return str;
}

function read_policies (res : string) : number[][] {
  const get_policy = (line : string) => {
    let p : number[] = [];
    const words : string[] = line.split(' ');
    for (const word of words) {
      if (word === "") {
        continue
      }
      const x : number = Number(word);
      console.assert(0 <= x && x <= 1);
      p.push(x);
    }
    return p;
  };
  const lines : string[] = res.split('\n').filter(x => (x !== ""));
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

const P1 = Team.unpack(
  'Fushigidane|Bulbasaur||-|SleepPowder,SwordsDance,RazorLeaf,BodySlam|||||||]' +
  'Hitokage|Charmander||-|FireBlast,FireSpin,Slash,Counter|||||||]' +
  'Zenigame|Squirtle||-|Surf,Blizzard,BodySlam,Rest|||||||]' +
  'Pikachuu|Pikachu||-|Thunderbolt,ThunderWave,Surf,SeismicToss|||||||]' +
  'Koratta|Rattata||-|SuperFang,BodySlam,Blizzard,Thunderbolt|||||||]' +
  'Poppo|Pidgey||-|DoubleEdge,QuickAttack,WingAttack,MirrorMove|||||||', Dex
)!.team;

const P2 = Team.unpack(
  'Kentarosu|Tauros||-|BodySlam,HyperBeam,Blizzard,Earthquake|||||||]' +
  'Rakkii|Chansey||-|Reflect,SeismicToss,SoftBoiled,ThunderWave|||||||]' +
  'Kabigon|Snorlax||-|BodySlam,Reflect,Rest,IceBeam|||||||]' +
  'Nasshii|Exeggutor||-|SleepPowder,Psychic,Explosion,DoubleEdge|||||||]' +
  'Sutaamii|Starmie||-|Recover,ThunderWave,Blizzard,Thunderbolt|||||||]' +
  'Fuudin|Alakazam||-|Psychic,SeismicToss,ThunderWave,Recover|||||||', Dex
)!.team;

const gens = new Generations(Dex);
const gen = gens.get(1);
const options = {
  p1: {name: 'Player A', team: P1},
  p2: {name: 'Player B', team: P2},
  seed: [1, 2, 3, 4],
  showdown: true,
  log: true,
};
let battle = Battle.create(gen, options);
// let result = battle.update(Choice.pass(), Choice.pass());
// const p1_actions = battle.choices('p1', result);
// console.log(p1_actions);
const input : string =  battle_bytes_string(battle, null);


const executablePath = './build/main'; 
let child = spawn(executablePath);
child.stdout.on('data', (data) => {
  const res : string = `${data}`;

  if (res[0] === '!') {
    console.log(res);
  } else {
    const policies : number[][] = read_policies(res);
    console.log(policies);
  }
});




child.stdin.write(input);
console.log(input);
// const log = new Log(gen, Lookup.get(gen), options);
// const display = () => {
//   for (const line of log.parse(battle.log!)) {
//     console.log(line);
//   }
// };

// const random = new Random();
// const choose = random.next.bind(random);

// // For convenience the engine actually is written so that passing in undefined
// // is equivalent to Choice.pass() but to appease the TypeScript compiler we're
// // going to be explicit here
// let result: Result, c1 = Choice.pass(), c2 = Choice.pass();
// while (!(result = battle.update(c1, c2)).type) {

//   display();

//   // special-cased choose method instead
//   c1 = battle.choose('p1', result, choose);
//   c2 = battle.choose('p2', result, choose);
// }
// // Remember to display any logs that were produced during the last update
// display();

// // The result is from the perspective of P1
// const msg = {
//   win: 'won by Player A',
//   lose: 'won by Player B',
//   tie: 'ended in a tie',
//   error: 'encountered an error',
// }[result.type];

// console.log(`Battle ${msg} after ${battle.turn} turns`);
