ids = list(range(1, 18))  # Assuming id_ is in the range 1-17
types = ["tag", "raw", "n1", "n2", "n3"]  # Known types
# types = ["tag", "raw", "n1", "n2", "n3", "swing_neigh_r1", "swing_neigh_r2", "swing_neigh_r3"]  # Known types
trials = list(range(5))  # Assuming trial is in the range 0-4

tags = {
    "trial": list(range(100)),
    "id": list(range(100)),
    "action": list(range(9)) + [88, 99],
    "fog": list(range(4)),
    "tray": [0, 1],
}