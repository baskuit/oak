import sys
import os
import random
import struct

def decode_u16(buffer, n):
    return buffer[n] + 256 * buffer[n + 1]
def decode_u4(buffer, n):
    return buffer[n] % 16, buffer[n] // 16

class Pokemon:
    n_moves = 166 # no None, Struggle
    n_species = 151
    n_status = 16
    n_types = 15

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
        self.species = buffer[21]
        self.type1, self.type2 = decode_u4(buffer, 22)
        self.level = buffer[23]
        self.sleep = None # duration info written after construction

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
        c += n_moves
        # status
        c += n_status
        # types
        t[c + self.type1] = 1.0
        t[c + self.type2] = 1.0
        c += n_types
        return t

class Volatiles:
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

    def to_tensor(self):
        return None

class Active:
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

    def to_tensor(self):
        return None

class Side:
    def __init__(self, buffer):
        self.active = Active(buffer[144 : 176])
        self.pokemon : list[Pokemon] = []
        for i in range(6):
            self.pokemon.append(Pokemon(buffer[i * 24 : (i + 1) * 24]))
        self.active_slot = buffer[176]

    def to_tensor(self):
        return None

class Battle:
    def __init__(self, buffer):
        self.p1 = Side(buffer[0 : 184])
        self.p2 = Side(buffer[184 : 368])
        self.result = None


class Frame:
    def __init__(self, buffer):
        self.battle = Battle(buffer[0 : 384])
        self.score = struct.unpack('<f', buffer[393 : 397])
        self.eval = struct.unpack('<f', buffer[397 : 401])
        self.iter = int.from_bytes(buffer[401 : 405])

    def to_tensor(self):
        return None       

FILENAME = 'buffer'
FRAME_SIZE = 405
NUM_READS = 1

def main():
    # Get the file size to determine how many frames fit
    filesize = os.path.getsize(FILENAME)
    # assert((filesize % FRAME_SIZE) != 0, "File size is not a multiple of the frame size, likely an error.")
    max_frames = filesize // FRAME_SIZE

    print(f"Found {max_frames} frames.")

    if max_frames == 0:
        print("File too small for even one 405-byte record.")
        return

    with open(FILENAME, 'rb') as f:
        for _ in range(NUM_READS):
            # Pick a random record index
            n = random.randint(0, max_frames - 1)
            offset = FRAME_SIZE * n

            # Seek to the offset and read the 405-byte slice
            f.seek(offset)
            slice_bytes = f.read(FRAME_SIZE)

            frame = Frame(slice_bytes)

if __name__ == "__main__":
    main()