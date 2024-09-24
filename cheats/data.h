#pragma once

#include <util.h>

#include "./random-set-data.h"

namespace RandomBattlesData {

using Helpers::Species;

static constexpr std::array<Species, 146> pokemonPool{
    Species::Bulbasaur,  Species::Ivysaur,    Species::Venusaur,
    Species::Charmander, Species::Charmeleon, Species::Charizard,
    Species::Squirtle,   Species::Wartortle,  Species::Blastoise,
    Species::Butterfree, Species::Beedrill,   Species::Pidgey,
    Species::Pidgeotto,  Species::Pidgeot,    Species::Rattata,
    Species::Raticate,   Species::Spearow,    Species::Fearow,
    Species::Ekans,      Species::Arbok,      Species::Pikachu,
    Species::Raichu,     Species::Sandshrew,  Species::Sandslash,
    Species::NidoranF,   Species::Nidorina,   Species::Nidoqueen,
    Species::NidoranM,   Species::Nidorino,   Species::Nidoking,
    Species::Clefairy,   Species::Clefable,   Species::Vulpix,
    Species::Ninetales,  Species::Jigglypuff, Species::Wigglytuff,
    Species::Zubat,      Species::Golbat,     Species::Oddish,
    Species::Gloom,      Species::Vileplume,  Species::Paras,
    Species::Parasect,   Species::Venonat,    Species::Venomoth,
    Species::Diglett,    Species::Dugtrio,    Species::Meowth,
    Species::Persian,    Species::Psyduck,    Species::Golduck,
    Species::Mankey,     Species::Primeape,   Species::Growlithe,
    Species::Arcanine,   Species::Poliwag,    Species::Poliwhirl,
    Species::Poliwrath,  Species::Abra,       Species::Kadabra,
    Species::Alakazam,   Species::Machop,     Species::Machoke,
    Species::Machamp,    Species::Bellsprout, Species::Weepinbell,
    Species::Victreebel, Species::Tentacool,  Species::Tentacruel,
    Species::Geodude,    Species::Graveler,   Species::Golem,
    Species::Ponyta,     Species::Rapidash,   Species::Slowpoke,
    Species::Slowbro,    Species::Magnemite,  Species::Magneton,
    Species::Farfetchd,  Species::Doduo,      Species::Dodrio,
    Species::Seel,       Species::Dewgong,    Species::Grimer,
    Species::Muk,        Species::Shellder,   Species::Cloyster,
    Species::Gastly,     Species::Haunter,    Species::Gengar,
    Species::Onix,       Species::Drowzee,    Species::Hypno,
    Species::Krabby,     Species::Kingler,    Species::Voltorb,
    Species::Electrode,  Species::Exeggcute,  Species::Exeggutor,
    Species::Cubone,     Species::Marowak,    Species::Hitmonlee,
    Species::Hitmonchan, Species::Lickitung,  Species::Koffing,
    Species::Weezing,    Species::Rhyhorn,    Species::Rhydon,
    Species::Chansey,    Species::Tangela,    Species::Kangaskhan,
    Species::Horsea,     Species::Seadra,     Species::Goldeen,
    Species::Seaking,    Species::Staryu,     Species::Starmie,
    Species::MrMime,     Species::Scyther,    Species::Jynx,
    Species::Electabuzz, Species::Magmar,     Species::Pinsir,
    Species::Tauros,     Species::Gyarados,   Species::Lapras,
    Species::Ditto,      Species::Eevee,      Species::Vaporeon,
    Species::Jolteon,    Species::Flareon,    Species::Porygon,
    Species::Omanyte,    Species::Omastar,    Species::Kabuto,
    Species::Kabutops,   Species::Aerodactyl, Species::Snorlax,
    Species::Articuno,   Species::Zapdos,     Species::Moltres,
    Species::Dratini,    Species::Dragonair,  Species::Dragonite,
    Species::Mewtwo,     Species::Mew};

constexpr bool isLevel100(Species species) noexcept {
  return (species == Species::Ditto) || (species == Species::Zubat);
}

}; // namespace RandomBattlesData