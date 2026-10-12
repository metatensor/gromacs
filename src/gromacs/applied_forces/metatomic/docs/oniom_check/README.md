# ONIOM link-atom check through the metatomic C API

An end-to-end check of `metatomic-oniom` / `metatomic-link-atoms` on the C-API
force provider, without libtorch: an analytic metatomic model
(`pair_model.cpp`) is loaded as a C-API plugin, and `gmx mdrun -rerun` is
compared against an independent evaluation and finite differences.

The system (`chain.top`, `chain.ndx`) is a 7-atom chain in a 3 nm box, wrapped
across the x boundary. The ML group (atoms 2-4) has four hydrogen caps, two of
them on the same MM atom (bonded to two ML atoms). The model's pair weights
depend on the element and on the `charge` input, so the cap element and the
cap charge (its MM atom's) both change the energy.

`check_oniom.py` checks the Metatomic energy against a Python evaluation of the
model on the ML atoms and caps, the total forces against finite differences
of the potential energy, and the virial against `0.5 dE/d(strain)` on
rectangular and sheared boxes.

## Build and run

```bash
# metatomic-core (libmetatomic) and its metatensor, e.g. from a CMake install
PREFIX=/path/to/metatomic/install
METATENSOR=/path/to/metatensor   # prefix with include/metatensor.h and lib/libmetatensor.so
c++ -std=c++17 -O2 -fPIC -shared pair_model.cpp -o libpair_model.so \
    -I$PREFIX/include -I$METATENSOR/include -I/path/to/nlohmann-json/include \
    -L$PREFIX/lib -L$METATENSOR/lib -lmetatomic -lmetatensor \
    -Wl,--disable-new-dtags,-rpath,$PREFIX/lib,-rpath,$METATENSOR/lib

python check_oniom.py --gmx build/bin/gmx --plugin libpair_model.so              # serial
python check_oniom.py --gmx build/bin/gmx --plugin libpair_model.so --ranks 2    # 2 x 1 x 1 DD
python check_oniom.py --gmx build/bin/gmx --plugin libpair_model.so --no-links   # no ONIOM
```

## Results (single precision, one workstation)

| configuration | Metatomic energy vs reference | max force error | virial |
| --- | --- | --- | --- |
| 4 caps, serial | 449.69269 / 449.69265 | 0.085 kJ/mol/nm (forces up to 1453) | all 6 components within 0.12 |
| 4 caps, 2 ranks | 449.69269 / 449.69265 | 0.111 | within 0.15 |
| no ONIOM, serial | 98.06195 / 98.06194 | 0.90 (forces up to 10803) | within 0.8 |
| no ONIOM, 2 ranks | 98.06194 / 98.06194 | 0.87 | within 0.8 |

The residuals are the finite-difference noise of single-precision energies.
This check found two bugs, fixed on this branch: the lattice shift of pairs in
sheared boxes (`box * n` instead of `box^T * n`), and missing backward ghosts
with two domain-decomposition cells. Without the cap virial correction, the
virial is off by up to 39 kJ/mol here (Vir-XX -1.13 instead of 37.75).

## ML region wider than rcoulomb

`check_wide_ml.py` runs a 6-atom ML rod (1.3 nm, `rod.top`, `rod.ndx`) among
four MM ions, with rcoulomb set just below the distance of ML atoms 0 and 3,
so the finite-difference and strain frames move that pair across the
cut-off. With PME it also checks that the Coulomb energy (SR + reciprocal)
matches a run with rcoulomb = 1.6 nm, where every ML pair is within the
cut-off.

```bash
python check_wide_ml.py --gmx build/bin/gmx --plugin libpair_model.so --coulomb pme
python check_wide_ml.py --gmx build/bin/gmx --plugin libpair_model.so --coulomb rf --ranks 2
```

| configuration | Coulomb energy vs rcoulomb = 1.6 nm | max force error | virial |
| --- | --- | --- | --- |
| PME, serial | -124.8170 / -124.8124 | 0.25 kJ/mol/nm (forces up to 161) | within 0.09 |
| PME, 2 ranks | -124.8147 / -124.8124 | 0.15 | within 0.05 |
| RF, serial | | 0.009 (forces up to 150) | within 0.003 |
| RF, 2 ranks | | 0.006 | within 0.007 |

Without the embedded Coulomb correction, the PME energy is off by 51 kJ/mol
(the ML pairs beyond rcoulomb keep their MM interaction), and with both PME
and RF the forces are off by 8000 kJ/mol/nm where the pair crosses the
cut-off.

## Several ML sites, with domain decomposition

`check_sites.py` runs four copies of the chain, each site the ML atoms of one copy with its
link caps, and gives each site its own `metatomic-site-charges` and
`metatomic-site-spin-multiplicities`. With a fifth parameter `kappa` in the model file, the pair
model requests the per-system `charge` and `spin_multiplicity` and scales the pairs of each
system by `1 + kappa (Q + 0.1 M)`, so a site given another site's values changes the energy.
Copy 0 wraps across the periodic boundary, copy 1 straddles a domain boundary, and copies 2
and 3 are within the model cutoff of each other, so their cross-site pairs must be left out.

With domain decomposition every rank holds every site, and each site is evaluated whole by
one rank, the largest sites first to the least-loaded rank. For each rank count the check
compares the energy with a site-by-site Python evaluation, the forces and virial with finite
differences, all three with the first rank count, and the site assignment logged by mdrun
with the expected line.

```bash
python check_sites.py --gmx build/bin/gmx --plugin libpair_model.so                     # 4 sites, 1-4 ranks, 2x2x1
python check_sites.py --gmx build/bin/gmx --plugin libpair_model.so --sites "0 1 2:12 3" --ranks 1 2 3
python check_sites.py --gmx build/bin/gmx --plugin libpair_model.so --sites "0 2:12" --ranks 1 3 4   # idle ranks
python check_sites.py --gmx build/bin/gmx --plugin libpair_model.so --system-energy
```

| sites | ranks | Metatomic energy (reference 2084.39157) | max force error | max virial error | vs 1 rank |
| --- | --- | --- | --- | --- | --- |
| 0 1 2 3 | 1 | 2084.39111 | 0.52 kJ/mol/nm (forces up to 3409) | 0.26 | |
| 0 1 2 3 | 2, 3, 4, 2x2x1 | 2084.39111 | 0.36-0.48 | 0.17-0.29 | forces within 1.4e-5 |

The site sizes count the caps: 7 model atoms for a full copy, 5 for `2:12`. Leaving copies
in no site gives ranks without sites ("rank 2: none"), which skip the model and still take
part in the force exchange. A site must lie within one molecule (grompp checks this), so
uneven sizes come from atom subsets rather than from joining copies.
