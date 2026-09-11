# Assumptions — issue #137

- **Where the pinned reference hashes live**: the issue asks for the solution file
  and the roster table to be "pinned by hash", without saying where. Resolved as:
  a committed `benchmarks/instances/mipfeas/references.csv` (`file,sha256,bytes`),
  mirroring `manifest.csv`'s shape. Because: a CSV pin table is a reviewable diff
  in the same style the directory already uses, and putting the hashes in Python
  constants would make an update a code change rather than a data change.
- **What `--verify` does about absent instances**: an absent instance could be
  read as a failure ("the roster is incomplete") or as a legitimate state.
  Resolved as: absent instances are counted and printed but do not fail `--verify`.
  Because: a checkout that fetched only the 11-instance smoke subset is normal,
  and the run driver already refuses a roster with missing instances before it
  starts — so making `--verify` fail on absence would only duplicate that check
  while making the common case noisy.
- **Two update flags rather than one**: resolved as `--update-references` (the
  yardstick: solution file + roster tables) and `--update-manifest` (instance
  bytes). Because: they have different blast radii — a moved reference invalidates
  every gap in the table at once, a moved instance invalidates one row — and a
  single flag would let an accepted instance re-fetch silently carry a yardstick
  revision with it.
