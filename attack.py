import numpy as np

plaintexts = []
timings = []

with open("aes_traces.csv", "r") as f:
    next(f)  # skip header
    for line in f:
        pt, t = line.strip().split(",")
        print("Plaintext:", pt, "Timing:", t)
        plaintexts.append(int(pt, 16))
        timings.append(int(t))

plaintexts = np.array(plaintexts, dtype=np.uint32)
timings = np.array(timings, dtype=np.uint32)

print("Loaded traces:", len(timings))
print("First plaintext:", hex(plaintexts[0]))
print("First timing:", timings[0])

