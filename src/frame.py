import sys
import os
import random
import struct

FRAME_SIZE = 442

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
        self.p1 = Duration(buffer[:4])
        self.p2 = Duration(buffer[4:8])

class Pokemon:
    n_moves = 164 # no None, Struggle
    n_status = 4 + 8 + 2 # 4 + 8 + 3
    n_types = 15

    n_dim = 5 + n_moves + n_status + n_types
    all_status = [0 for _ in range(n_status)]
    rest_durations = [0 for _ in range(12)]


    def __init__(self, buffer):
        assert(len(buffer) == 24)
        self.hp = decode_u16(buffer, 0)
        self.atk = decode_u16(buffer, 2)
        self.def_ = decode_u16(buffer, 4)
        self.spe = decode_u16(buffer, 6)
        self.spc = decode_u16(buffer, 8)
        self.moves = []
        for i in range(4):
            self.moves.append(
                [buffer[10 + 2 * i], buffer[11 + 2 * i]]
            )

        self.current_hp = decode_u16(buffer, 18)
        self.status = buffer[20]
        self.is_sleep: bool = (self.status & 7)
        self.is_self = self.status >> 7
        self.species = buffer[21]
        self.type1, self.type2 = decode_u4(buffer, 22)
        self.level = buffer[23]
        self.sleep_duration = None # duration info written after construction
        # self.si = self.status_index()

    def status_index(self, frame_number):
        status_index = -1
        if not self.status:
            return status_index 
        if not self.is_sleep:
            status_index = lsb(self.status) - 4
            assert(status_index >= 0 and status_index < 4)
        else:
            if not self.is_self:
                status_index = 4 + (self.sleep_duration)
                # print("{:08b}".format(self.status), self.sleep_duration)
                assert(status_index >= 4 and status_index < 12)
            else:
                # if (self.sleep_duration > 0):
                #     print("weird shit at :", frame_number)
                # print(self.sleep_duration)
                s = self.status & 7
                self.rest_durations[s] += 1
                status_index = 12 + (s - 1)
                assert(status_index >= 12 and status_index < 14)
        self.all_status[status_index] += 1
        return status_index

    def to_tensor(self, t):
        c = 0
        # stats
        t[0] = self.hp
        t[1] = self.atk
        t[2] = self.def_
        t[3] = self.spc
        t[4] = self.spe
        c += 5
        # moves
        for i in range(4):
            m = self.moves[i][0]
            pp = self.moves[i][1]
            if (m != 0):
                t[c + (m - 1)] = 1
            assert(m < 165 and m > 0)
        c += n_moves
        # status
        if self.status:
            t[c + self.status_index()] = 1

        c += n_status
        # types
        t[c + self.type1] = 1.0
        t[c + self.type2] = 1.0
        c += n_types
        return t

class Volatiles:
    n_confusion = 5
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
        self.confusion = (bits >> 18) & 0b111
        self.attacks = (bits >> 21) & 0b111
        self.state = (bits >> 24) & 0xFFFF
        self.substitute_hp = (bits >> 40) & 0xFF
        self.transform_id = (bits >> 48) & 0xF
        self.disable_duration = (bits >> 52) & 0xF
        self.disable_move = (bits >> 56) & 0b111
        self.toxic_counter = (bits >> 59) & 0b11111
        self.confusion_duration = None

    def to_tensor(self, t):
        t[0] = self.binding
        t[1] = self.substitute
        t[2] = self.recharging
        t[3] = self.leech_seed
        t[4] = self.toxic
        t[5] = self.light_screen
        t[6] = self.reflect
        t[7] = self.substitute_hp
        t[8] = self.toxic_counter
        t[8 + self.confusion_duration] = 1

class Active:
    n_dim = 5 + Volatiles.n_dim

    def __init__(self, buffer):
        self.hp = decode_u16(buffer, 0)
        self.atk = decode_u16(buffer, 2)
        self.def_ = decode_u16(buffer, 4)
        self.spe = decode_u16(buffer, 6)
        self.spc = decode_u16(buffer, 8)
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
        t[0] = self.hp
        t[1] = self.atk
        t[2] = self.def_
        t[3] = self.spe
        t[4] = self.spc
        self.volatiles.to_tensor(t[5:])

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

def print_status_duration(side, duration):
    for _ in range(6):
        i = side.order[_] - 1
        pokemon = side.pokemon[i]
        print(_, "{:08b}".format(pokemon.status), duration.sleeps[_])


class Frame:
    def __init__(self, buffer, frame_index):
        self.battle = Battle(buffer[0 : 384])
        self.durations = Durations(buffer[384 : 392])

        for _ in range(6):
            i = self.battle.p1.order[_] - 1
            j = self.battle.p2.order[_] - 1
            self.battle.p1.pokemon[i].sleep_duration = self.durations.p1.sleeps[_]
            self.battle.p2.pokemon[j].sleep_duration = self.durations.p2.sleeps[_]
        for _ in range(6):
            self.battle.p1.pokemon[_].status_index(frame_index)
            self.battle.p2.pokemon[_].status_index(frame_index)

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
        print("File too small for even one 405-byte record.")
        return

    with open(FILENAME, 'rb') as f:
        for _ in range(max_frames):
            f.seek(_ * FRAME_SIZE)
            slice_bytes = f.read(FRAME_SIZE)

            frame = Frame(slice_bytes, _)

    print(Pokemon.all_status)
    print(Pokemon.rest_durations)
if __name__ == "__main__":
    main()