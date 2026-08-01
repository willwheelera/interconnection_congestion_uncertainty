import numpy as np

NGEN = 4
N = int(2**NGEN)

def neighbors(ids):
    # avail position k is the (NGEN-1-k)-th bit of the integer encoding
    bit_mask = (1 << (NGEN - 1 - np.arange(NGEN))).astype(np.int64)
    neighbor_ids = ids[:, None] ^ bit_mask[None, :]                 # (n, NGEN)
    return neighbor_ids

def print_binary(y):
    l = [np.binary_repr(x, width=4) for x in y.ravel()]
    s = np.array(l).reshape(y.shape)
    print(s)

np.random.seed(0)
ids = np.arange(N)
a = np.random.random(N)
uphill = np.arange(N)

nb = neighbors(ids)
for n in range(N):
    i = np.argmax(a[nb[n]])
    print(n, i, nb[n, i])
    if a[nb[n, i]] > a[n]:
        uphill[n] = nb[n, i]

nb_table = np.concatenate([ids[:, np.newaxis], nb], axis=1)
print(nb_table)
print(a[nb_table].round(2))
#print(np.stack([ids, a]).T.round(2))
b = np.stack([ids, uphill], axis=1)
print(b)

print_binary(b)

climb = [ids, uphill]
for i in range(5):
    climb.append(uphill[climb[-1]])
climb = np.array(climb).T
print(climb)

final = climb[:, [0, -1]]
print_binary(final)
