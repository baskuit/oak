pkmn.ai covers all published mons bots and they all suggest one thing: search is needed more than anything else.

search requires many creations, transitions, and evaluation of a battle. `pokemon-showdown` has been the best hope for most devs, but its still orders of magnitude slower than `libpkmn`.

search in an imperfect information game like OU singles is also prohibitively expensive in its own right. It is an important motivator for search in *perfect information* battles. 

Certainly mastery of the former requires mastery of the latter. And battles are always converging to a state of perfect information as sets are revealed during play.

that is the purpose of this repo. To combine the `libpkmn` simulator and `pinyon` search library