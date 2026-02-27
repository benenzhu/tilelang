

def get_val2(tx):
    vec = 0
    i = 0
    a = [0, ((i * 8 + vec) // 8 * 64 + tx // 8) // 8, ((i * 8 + vec) // 8 * 64 + tx // 8) % 8 * 64 + ((tx % 8 * 8 + (i * 8 + vec) % 8) // 32 + ((i * 8 + vec) // 8 * 64 + tx // 8) % 8 // 4) % 2 * 32 + ((tx % 8 * 8 + (i * 8 + vec) % 8) % 32 // 16 + ((i * 8 + vec) // 8 * 64 + tx // 8) % 4 // 2) % 2 * 16 + ((tx % 8 * 8 + (i * 8 + vec) % 8) % 16 // 8 + ((i * 8 + vec) // 8 * 64 + tx // 8) % 2) % 2 * 8 + (tx % 8 * 8 + (i * 8 + vec) % 8) % 8]
    return a[1] * 512 + a[2]
def get_val(tx):
    return (tx//8)*64 + tx%8*8
for tx in range(64):
    print(tx, get_val(tx), get_val(tx) - get_val(tx - 1))