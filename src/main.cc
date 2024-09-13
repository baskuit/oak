#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <pkmn.h>
#include <sides.h>
#include <types/types.h>

pkmn_choice choose(pkmn_gen1_battle *battle, pkmn_psrng *random,
                   pkmn_player player, pkmn_choice_kind request,
                   pkmn_choice choices[]) {
  uint8_t n = pkmn_gen1_battle_choices(battle, player, request, choices,
                                       PKMN_CHOICES_SIZE);
  // Technically due to Generation I's Transform + Mirror Move/Metronome PP
  // error if the battle contains Pokémon with a combination of Transform,
  // Mirror Move/Metronome, and Disable its possible that there are no
  // available choices (softlock), though this is impossible here given that
  // our example battle involves none of these moves
  assert(n > 0);
  // pkmn_gen1_battle_choices determines what the possible choices are - the
  // simplest way to choose an option here is to just use the PSRNG to pick one
  // at random
  return choices[(uint64_t)pkmn_psrng_next(random) * n / 0x100000000];
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <seed>\n", argv[0]);
    return 1;
  }

  // Expect that we have been given a decimal seed as our only argument
  char *end = NULL;
  uint64_t seed = strtoul(argv[1], &end, 10);
  if (errno) {
    fprintf(stderr, "Invalid seed: %s\n", argv[1]);
    fprintf(stderr, "Usage: %s <seed>\n", argv[0]);
    return 1;
  }

  // We could use C's srand() and rand() function but for point of example
  // we will demonstrate how to use the PSRNG that is exposed by libpkmn
  pkmn_psrng random;
  pkmn_psrng_init(&random, seed);
  // Preallocate a small buffer for the choice options throughout the battle
  pkmn_choice choices[PKMN_CHOICES_SIZE];

  // libpkmn doesn't provide any helpers for initializing the battle structure
  // (the library is intended to be wrapped by something with a higher level
  // API). This setup borrows the serialized state of the setup from the Zig
  // example, though will end up with a different result because it's using a
  // different RNG
  pkmn_gen1_battle battle;
  memcpy(battle.bytes, sides[0], 184);
  memcpy(battle.bytes + 184, sides[1], 184);
  memset(battle.bytes + 2 * 184, 0, 384 - 2 * 184);
  // Preallocate a buffer for protocol message logs - PKMN_LOGS_SIZE is
  // guaranteed to be large enough for a single update. This will only be
  // written to if -Dlog is enabled - NULL can be used to turn all of the
  // logging into no-ops
  uint8_t buf[PKMN_LOGS_SIZE];
  pkmn_gen1_log_options log_options = {.buf = buf, .len = PKMN_LOGS_SIZE};

  // Initialize the battle options with the log options (in this example
  // the chance and calc features are not demonstrated and thus we simply
  // pass NULL). If no optional features are desired (or if they weren't
  // enabled at compile time) the options struct is unnecessary - we can
  // simply pass NULL as the last argument to the update function
  pkmn_gen1_battle_options options;
  pkmn_gen1_battle_options_set(&options, &log_options, NULL, NULL);

  pkmn_result result;
  // Pedantically these *should* be pkmn_choice_init(PKMN_CHOICE_PASS, 0), but
  // libpkmn commits to always ensuring the pass choice is 0 so we can simplify
  // things here
  pkmn_choice c1 = 0, c2 = 0;
  // We're also taking advantage of the fact that the PKMN_RESULT_NONE is
  // guaranteed to be 0, so we don't actually need to check against it here
  while (!pkmn_result_type(
      result = pkmn_gen1_battle_update(&battle, c1, c2, &options))) {
    // If -Dlog is enabled we would now do something with the data in `buf`
    // before coming up with our next set of choices
    c1 = choose(&battle, &random, PKMN_PLAYER_P1, pkmn_result_p1(result),
                choices);
    c2 = choose(&battle, &random, PKMN_PLAYER_P2, pkmn_result_p2(result),
                choices);
    // We need to reset our options struct after each update, in this case
    // to reset the log stream before it gets written to again. The middle
    // two arguments (for log and chance, respectively) should almost always
    // be NULL for resets, but the final argument may be used for calc
    // overrides. If you have no battle options then this can be skipped
    pkmn_gen1_battle_options_set(&options, NULL, NULL, NULL);
  }
  // The only error that can occur is if we didn't provide a large enough
  // buffer, but PKMN_MAX_LOGS is guaranteed to be large enough so errors here
  // are impossible. Note however that this is tracking a different kind of
  // error than PKMN_RESULT_ERROR
  assert(!pkmn_error(result));

  // The battle is written in native endianness so we need to do a bit-hack to
  // figure out the system's endianess before we can read the 16-bit turn data
  volatile uint32_t endian = 0x01234567;
  uint16_t turns = (*((uint8_t *)(&endian))) == 0x67
                       ? battle.bytes[368] | battle.bytes[369] << 8
                       : battle.bytes[368] << 8 | battle.bytes[369];

  // The result is from the perspective of P1
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_WIN: {
    printf("Battle won by Player A after %d turns\n", turns);
    break;
  }
  case PKMN_RESULT_LOSE: {
    printf("Battle won by Player B after %d turns\n", turns);
    break;
  }
  case PKMN_RESULT_TIE: {
    printf("Battle ended in a tie after %d turns\n", turns);
    break;
  }
  case PKMN_RESULT_ERROR: {
    printf("Battle encountered an error after %d turns\n", turns);
    break;
  }
  default:
    assert(false);
  }

  return 0;
}
