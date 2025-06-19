import sys
import os
import struct

FRAME_SIZE = 442
ACTIVE_OUT = 56
POKEMON_OUT = 40 
assert(ACTIVE_OUT + 5 * POKEMON_OUT == 256)

def decode_u16(buffer, n):
    return buffer[n] + 256 * buffer[n + 1]
def decode_u4(buffer, n):
    return buffer[n] % 16, buffer[n] // 16
def lsb(x):
    return (x & -x).bit_length()

class Duration:
    def __init__(self, buffer):
        s = int.from_bytes(buffer[0 : 3], byteorder="little")
        self.sleeps = []
        for _ in range(6):
            self.sleeps.append(s & 7)
            s >>= 3
        self.confusion = (buffer[2] >> 2) & 7
        self.disable = (buffer[2] >> 5) + 8 * (buffer[3] & 1)
        self.attacking = (buffer[3] >> 1) & 7
        self.binding = (buffer[3] >> 4) & 7

class Durations:
    def __init__(self, buffer):
        assert(len(buffer) == 8)
        self.p1 = Duration(buffer[0:4])
        self.p2 = Duration(buffer[4:8])

class Stats():
    def __init__(self, buffer):
        self.hp = decode_u16(buffer, 0)
        self.atk = decode_u16(buffer, 2)
        self.def_ = decode_u16(buffer, 4)
        self.spc = decode_u16(buffer, 6)
        self.spe = decode_u16(buffer, 8)

    def to_tensor(self, t):
        assert(t.shape[0] == 5)
        t[0] = self.hp / 999
        t[1] = self.atk / 999
        t[2] = self.def_ / 999
        t[3] = self.spc / 999
        t[4] = self.spe / 999

class Pokemon:
    n_moves = 164 # no None, Struggle
    n_status = 4 + 8 + 2 # non sleep, slept, resting
    n_types = 15
    n_dim = 5 + n_moves + n_status + n_types

    def __init__(self, buffer):
        assert(len(buffer) == 24)
        self.stats = Stats(buffer)
        self.moves = []
        for i in range(4):
            m = buffer[10 + 2 * i]
            pp = buffer[11 + 2 * i]
            assert(m >= 1 and m < 165)
            self.moves.append([m, pp])

        self.current_hp = decode_u16(buffer, 18)
        self.status = buffer[20]
        self.is_sleep = (self.status & 7)
        self.is_self = self.status >> 7
        self.species = buffer[21]
        self.type1, self.type2 = decode_u4(buffer, 22)
        self.level = buffer[23]
        self.sleep_duration = None # duration info written after construction

    def status_index(self):
        status_index = -1
        if not self.status:
            return status_index 
        if not self.is_sleep:
            status_index = lsb(self.status) - 4
            assert(status_index >= 0 and status_index < 4)
        else:
            if not self.is_self:
                status_index = 4 + (self.sleep_duration)
                assert(status_index >= 4 and status_index < 12)
            else:
                s = self.status & 7
                status_index = 12 + (s - 1) #kinda out of order but doesnt matter   
                assert(status_index >= 12 and status_index < 14)
        return status_index

    def to_tensor(self, t, write_stats=True):
        assert(t.shape[0] == self.n_dim)
        if write_stats:
            self.stats.to_tensor(t[0: 5])
        c = 5
        for i in range(4):
            m, pp = self.moves[i]
            if m and pp:
                t[c + (m - 1)] = 1.0
        c += self.n_moves
        if self.status:
            t[c + self.status_index()] = 1.0
        c += self.n_status
        t[c + self.type1] = 1.0
        t[c + self.type2] = 1.0

class Volatiles:
    n_confusion = 6
    n_dim = 9 + n_confusion # ls/reflect/trapping/recharge/leech/toxic/toxic_counter/sub/subhp

    def __init__(self, buffer: bytes):
        assert len(buffer) == 8
        bits = int.from_bytes(buffer, byteorder="little")
        self.bide         = bool((bits >> 0)  & 1)
        self.thrashing     = bool((bits >> 1)  & 1)
        self.multi_hit     = bool((bits >> 2)  & 1)
        self.flinch        = bool((bits >> 3)  & 1)
        self.charging      = bool((bits >> 4)  & 1)
        self.binding       = bool((bits >> 5)  & 1)
        self.invulnerable  = bool((bits >> 6)  & 1)
        self.confusion     = bool((bits >> 7)  & 1)
        self.mist          = bool((bits >> 8)  & 1)
        self.focus_energy  = bool((bits >> 9)  & 1)
        self.substitute    = bool((bits >> 10) & 1)
        self.recharging    = bool((bits >> 11) & 1)
        self.rage          = bool((bits >> 12) & 1)
        self.leech_seed    = bool((bits >> 13) & 1)
        self.toxic         = bool((bits >> 14) & 1)
        self.light_screen  = bool((bits >> 15) & 1)
        self.reflect       = bool((bits >> 16) & 1)
        self.transform     = bool((bits >> 17) & 1)
        self.confusion_raw = (bits >> 18) & 0b111
        self.attacks = (bits >> 21) & 0b111
        self.state = (bits >> 24) & 0xFFFF
        self.substitute_hp = (bits >> 40) & 0xFF
        self.transform_id = (bits >> 48) & 0xF
        self.disable_duration = (bits >> 52) & 0xF
        self.disable_move = (bits >> 56) & 0b111
        self.toxic_counter = (bits >> 59) & 0b11111
        self.confusion_duration = None

    def to_tensor(self, t):
        assert(t.shape[0] == self.n_dim)
        t[0] = self.binding
        t[1] = self.substitute
        t[2] = self.recharging
        t[3] = self.leech_seed
        t[4] = self.toxic
        t[5] = self.light_screen
        t[6] = self.reflect
        t[7] = self.substitute_hp / 176 # chansey hp / 4
        t[8] = self.toxic_counter / 16 # if its at 16 it ohkos yeah?
        t[9 + self.confusion_duration] = 1

class Active:
    n_dim = Pokemon.n_dim + Volatiles.n_dim

    def __init__(self, buffer):
        self.stats = Stats(buffer[0 : 10])
        self.species = buffer[10]
        self.type1, self.type2 = decode_u4(buffer, 11)
        self.boost_atk, self.boost_def = decode_u4(buffer, 12)
        self.boost_spe, self.boost_spc = decode_u4(buffer, 13)
        self.boost_acc, self.boost_eva = decode_u4(buffer, 14)
        self.volatiles = Volatiles(buffer[16 : 24])
        self.moves = []
        for i in range(4):
            self.moves.append(
                [buffer[24 + 2 * i], buffer[25 + 2 * i]]
            )

    def to_tensor(self, t):
        assert(t.shape[0] == self.n_dim)
        self.stats.to_tensor(t[0 : 5])
        self.volatiles.to_tensor(t[Pokemon.n_dim : Active.n_dim])

class Side:
    def __init__(self, buffer):
        self.pokemon : list[Pokemon] = []
        for i in range(6):
            self.pokemon.append(Pokemon(buffer[i * 24 : (i + 1) * 24]))
        self.active = Active(buffer[144 : 176])
        self.order = []
        for i in range(6):
            self.order.append(buffer[176 + i])
        self.last_selected_move = buffer[182]
        self.last_used_move = buffer[183]

class Battle:
    def __init__(self, buffer):
        self.p1 = Side(buffer[0 : 184])
        self.p2 = Side(buffer[184 : 368])
        self.result = None

class Frame:
    active_dim = Active.n_dim
    pokemon_dim = Pokemon.n_dim

    def __init__(self, buffer):
        self.battle = Battle(buffer[0 : 384])
        self.durations = Durations(buffer[384 : 392])

        self.battle.p1.active.volatiles.confusion_duration = self.durations.p1.confusion
        self.battle.p2.active.volatiles.confusion_duration = self.durations.p2.confusion

        for _ in range(6):
            i = self.battle.p1.order[_] - 1
            j = self.battle.p2.order[_] - 1
            self.battle.p1.pokemon[i].sleep_duration = self.durations.p1.sleeps[_]
            self.battle.p2.pokemon[j].sleep_duration = self.durations.p2.sleeps[_]

        self.result = int(buffer[392])
        self.eval = struct.unpack('<f', buffer[393 : 397])[0]
        self.score = struct.unpack('<f', buffer[397 : 401])[0]
        self.iter = decode_u16(buffer, 401) + (256 ** 2) * decode_u16(buffer, 403)
        row_col = int(buffer[405])
        self.m = (row_col // 9) + 1
        self.n = (row_col % 9) + 1
        self.p1_visits = []
        for _ in range(self.m):
            self.p1_visits.append(decode_u16(buffer, 406 + 2*_))
        self.p2_visits = []
        for _ in range(self.n):
            self.p2_visits.append(decode_u16(buffer, 424 + 2*_))
        self.p1_policy = [x / self.iter for x in self.p1_visits]
        self.p2_policy = [x / self.iter for x in self.p2_visits]

    def write_to_buffers(self, buffers, index):
        p, a, s, e, acc = buffers.to_tensor(index)

        for i, side in enumerate([self.battle.p1, self.battle.p2]):
            active_pokemon = side.pokemon[side.order[0] - 1]
            active_pokemon.to_tensor(a[i, 0, 0 : Pokemon.n_dim], write_stats=False)
            side.active.to_tensor(a[i, 0])

            hp_percent = active_pokemon.current_hp / active_pokemon.stats.hp
            assert(hp_percent >= 0 and hp_percent <= 1)
            acc[i, 0, 0] = hp_percent
            for k in range(5):
                pokemon = side.pokemon[side.order[k + 1] - 1]
                pokemon.to_tensor(p[i, k])
                hp_percent = pokemon.current_hp / pokemon.stats.hp
                assert(hp_percent >= 0 and hp_percent <= 1)
                acc[i, 0, ACTIVE_OUT + k * POKEMON_OUT] = hp_percent
        
        s[0] = self.score
        e[0] = self.eval

def main():

    if len(sys.argv) < 2:
        print("Error: provide path to buffer.")
        exit()
        
    FILENAME = sys.argv[1]
    filesize = os.path.getsize(FILENAME)
    max_frames = filesize // FRAME_SIZE

    if len(sys.argv) >= 3:
        max_frames = min(max_frames, int(sys.argv[2]))

    print(f"Reading {max_frames} frames.")

    if max_frames == 0:
        print("File too small for even one record.")
        return

    with open(FILENAME, 'rb') as f:
        for _ in range(max_frames):
            f.seek(_ * FRAME_SIZE)
            slice_bytes = f.read(FRAME_SIZE)

            frame = Frame(slice_bytes, _)

def print_dims():
    print(Pokemon.n_dim)
    print(Active.n_dim)
    

if __name__ == "__main__":
    print_dims()