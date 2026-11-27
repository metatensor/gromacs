#!/usr/bin/env python3
"""Run metatomic-core's Lennard-Jones model through grompp/mdrun and check energy and forces.

Usage: python3 metatomic-capi-smoke.py /absolute/path/to/gmx /absolute/path/to/metatomic-lj-plugin.so

The plugin is installed with libmetatomic, in <prefix>/libexec/metatomic.

Serial runs check the model in nm and in Angstrom. Two-rank runs with domain
decomposition (each atom on its own rank, so the pair crosses ranks) check a 3D
periodic box and an xy-periodic slab whose pair is farther apart in z than half
the box height, which must not be shifted along z. Four-rank runs check boxes of
64 and 1000 atoms, and domains of 0.7 nm, narrower than the 0.9 nm range of the
model, so that atoms come from two domains away, and ranks without atoms.
ONIOM runs check, on one and two ranks, that metatomic-oniom removes the PME
Coulomb interaction of two embedded charges 1.2 nm apart at any distance: their
Coulomb energy is the same with the pair beyond rcoulomb (0.9 nm) and within it
(1.3 nm), and differs by about 29 kJ/mol without ONIOM.
They use thread-MPI, or the launcher in GMX_MPIEXEC with a library-MPI build,
e.g. GMX_MPIEXEC="mpirun -np".
The classical Lennard-Jones parameters are zero, so only the plugin can produce
the expected energy and forces. Only the Python standard library and the built
GROMACS executable are needed.
"""
import itertools
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

SIGMA, EPSILON, CUTOFF = 0.34, 0.99, 0.9  # Lennard-Jones parameters of the model, nm and kJ/mol
NM = {'nm': 1, 'Angstrom': 10}  # lengths in nm, in the length unit of the model

TOPOLOGY = '''[ defaults ]
1 2 no 1 1
[ atomtypes ]
; no classical LJ: the energy and forces must come from the plugin
Ar 18 39.948 0.0 A 0.0 0.0
[ moleculetype ]
ARG 1
[ atoms ]
1 Ar 1 ARG Ar 1 0.0 39.948
[ system ]
LJ smoke test
[ molecules ]
ARG {count}
'''


def run(command, cwd, input=None):
    result = subprocess.run(command, cwd=cwd, input=input, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))} failed:\n{result.stdout}")
    return result.stdout


def rows(path):
    return [[float(x) for x in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(('#', '@'))]


def mpi_library(gmx):
    """'thread_mpi', 'MPI' or 'none', from gmx -version"""
    for line in run([gmx, '-quiet', '-version'], cwd='.').splitlines():
        if line.strip().startswith('MPI library:'):
            return line.split(':', 1)[1].strip()
    return 'none'


def expected(x, box, periodic):
    """Lennard-Jones energy and the force on each atom, minimum image along periodic axes"""
    energy, forces = 0.0, [[0.0] * 3 for _ in x]
    for i, j in itertools.combinations(range(len(x)), 2):
        d = [b - a for a, b in zip(x[i], x[j])]
        d = [v - L * round(v / L) if p else v for v, L, p in zip(d, box, periodic)]
        r = math.sqrt(sum(v * v for v in d))
        assert abs(r - CUTOFF) > 1e-3, 'a pair this close to the cutoff makes the test fragile'
        if r >= CUTOFF:
            continue
        sr6 = (SIGMA / r) ** 6
        cr6 = (SIGMA / CUTOFF) ** 6
        # the energy of metatomic-core's model is shifted to zero at the cutoff
        energy += 4 * EPSILON * (sr6 * sr6 - sr6) - 4 * EPSILON * (cr6 * cr6 - cr6)
        derivative = 4 * EPSILON * (-12 * sr6 * sr6 + 6 * sr6) / r
        for axis in range(3):
            forces[j][axis] -= derivative * d[axis] / r
            forces[i][axis] += derivative * d[axis] / r
    return energy, forces


def lattice(counts, box):
    """Argon on a lattice of spacing 0.7 nm, moved by up to 0.03 nm: no pair near the cutoff"""
    return [tuple(round((c + 0.5) * L / n + 0.03 * math.sin(1.3 * i + 2.1 * j + 3.7 * k + 1.9 * axis), 3)
                  for axis, (c, L, n) in enumerate(zip((i, j, k), box, counts)))
            for i, j, k in itertools.product(*(range(n) for n in counts))]


def mdrun_command(gmx, mdrun, dd, launcher, library):
    """mdrun on the ranks of the domain decomposition grid dd"""
    ranks = math.prod(dd)
    if ranks > 1:
        mdrun = [*mdrun, '-dd', *map(str, dd), '-npme', '0']
    if launcher:
        return [*launcher, str(ranks), gmx, *mdrun]
    if library == 'thread_mpi':
        return [gmx, *mdrun, '-ntmpi', str(ranks)]
    return [gmx, *mdrun]


ONIOM_TOPOLOGY = '''[ defaults ]
1 2 no 1 1
[ atomtypes ]
Ar 18 39.948 0.0 A 0.0 0.0
[ moleculetype ]
PAIR 1
[ atoms ]
1 Ar 1 PAIR A1 1  0.5 39.948
2 Ar 1 PAIR A2 1 -0.5 39.948
[ system ]
two embedded charges
[ molecules ]
PAIR 1
'''


def oniom_coulomb(gmx, plugin, root, launcher, library):
    """PME Coulomb energy of two embedded charges 1.2 nm apart, with and without ONIOM"""
    (root / 'oniom.top').write_text(ONIOM_TOPOLOGY)
    atoms = ''.join(f'{1:5d}{"PAIR":<5s}{name:>5s}{i + 1:5d}{x:8.3f}{2:8.3f}{2:8.3f}\n'
                    for i, (name, x) in enumerate([('A1', 1.4), ('A2', 2.6)]))
    (root / 'oniom.gro').write_text(f'pair\n2\n{atoms}   4.00000   4.00000   4.00000\n')

    def coulomb(oniom, cutoff, dd):
        name = f'oniom-{oniom}-{cutoff}-{math.prod(dd)}'
        (root / f'{name}.mdp').write_text(f'''integrator = md
nsteps = 0
cutoff-scheme = Verlet
verlet-buffer-tolerance = -1
rlist = {cutoff}
coulombtype = PME
rcoulomb = {cutoff}
rvdw = {cutoff}
comm-mode = none
metatomic-active = true
metatomic-input-group = System
metatomic-model = metatomic-lj-model
metatomic-plugin = {plugin}
metatomic-model-options = sigma={SIGMA} epsilon={EPSILON} cutoff={CUTOFF} atomic_type=18 length_unit=nm energy_unit="kJ/mol"
metatomic-oniom = {oniom}
''')
        run([gmx, 'grompp', '-f', f'{name}.mdp', '-c', 'oniom.gro', '-p', 'oniom.top',
             '-o', f'{name}.tpr'], cwd=root)
        run(mdrun_command(gmx, ['mdrun', '-s', f'{name}.tpr', '-deffnm', name, '-nb', 'cpu'],
                          dd, launcher, library), cwd=root)
        log = (root / f'{name}.log').read_text()
        assert ('removing the classical Coulomb interaction' in log) == (oniom == 'yes'), name
        if math.prod(dd) > 1:
            assert 'Domain decomposition grid {} x {} x {}'.format(*dd) in log, f'{name}: no DD'
        run([gmx, 'energy', '-f', f'{name}.edr', '-o', f'{name}-energy.xvg'], cwd=root,
            input='Coulomb-(SR)\nCoul.-recip.\n0\n')
        return sum(rows(root / f'{name}-energy.xvg')[0][1:])

    for dd in [(1, 1, 1), (2, 1, 1)]:
        ranks = math.prod(dd)
        if ranks > 1 and not launcher and library != 'thread_mpi':
            print(f'oniom ({ranks} ranks): SKIP (needs thread-MPI, or GMX_MPIEXEC with an MPI build)')
            continue
        beyond, within = coulomb('yes', 0.9, dd), coulomb('yes', 1.3, dd)
        assert abs(beyond - within) < 0.1, ('oniom', ranks, beyond, within)
        if ranks == 1:
            # without ONIOM, the pair beyond rcoulomb keeps its reciprocal-space interaction
            assert abs(coulomb('no', 0.9, dd) - within) > 10, 'the ONIOM check cannot fail'
        print(f'oniom ({ranks} rank{"s" if ranks > 1 else ""}): Coulomb energy independent of rcoulomb PASS')


def main():
    gmx, plugin = (str(Path(p).resolve(strict=True)) for p in sys.argv[1:])
    launcher = shlex.split(os.environ.get('GMX_MPIEXEC', ''))
    library = mpi_library(gmx)
    cases = [
        # name, model length unit, pbc, box, positions, domain decomposition grid
        ('nm', 'nm', 'xyz', (3, 3, 3), [(0.3, 0.4, 0.5), (0.7, 0.4, 0.5)], (1, 1, 1)),
        ('angstrom', 'Angstrom', 'xyz', (3, 3, 3), [(0.3, 0.4, 0.5), (0.7, 0.4, 0.5)], (1, 1, 1)),
        ('dd-xyz', 'Angstrom', 'xyz', (3, 3, 3), [(1.3, 0.4, 0.5), (1.7, 0.45, 0.6)], (2, 1, 1)),
        # dz = 0.65 nm is more than half the 1.2 nm slab, but z is not periodic
        ('dd-xy-slab', 'nm', 'xy', (3, 3, 1.2), [(1.3, 0.4, 0.2), (1.7, 0.4, 0.85)], (2, 1, 1)),
        ('dd-64', 'Angstrom', 'xyz', (2.8, 2.8, 2.8), lattice((4, 4, 4), (2.8, 2.8, 2.8)), (2, 2, 1)),
        ('dd-1000', 'nm', 'xyz', (7, 7, 7), lattice((10, 10, 10), (7, 7, 7)), (2, 2, 1)),
        ('dd-thin', 'nm', 'xyz', (2.8, 2.8, 2.8), lattice((4, 4, 4), (2.8, 2.8, 2.8)), (4, 1, 1)),
        # two of the four ranks have no atoms, and send and receive nothing
        ('dd-empty-ranks', 'nm', 'xyz', (3, 3, 3), [(1.3, 0.4, 0.5), (1.7, 0.45, 0.6)], (4, 1, 1)),
    ]
    # pbc = xy needs walls with Verlet lists; zero LJ parameters make them exert no force
    walls = 'nwall = 2\nwall-type = 9-3\nwall-atomtype = Ar Ar\nwall-density = 1 1\n'
    with tempfile.TemporaryDirectory(prefix='gromacs-capi-') as temporary:
        root = Path(temporary)
        for name, unit, pbc, box, x, dd in cases:
            ranks = math.prod(dd)
            if ranks > 1 and not launcher and library != 'thread_mpi':
                print(f'{name}: SKIP (needs thread-MPI, or GMX_MPIEXEC with an MPI build)')
                continue
            atoms = ''.join(f'{i + 1:5d}ARG     Ar{i + 1:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n'
                            for i, p in enumerate(x))
            (root / f'{name}.top').write_text(TOPOLOGY.format(count=len(x)))
            (root / f'{name}.gro').write_text(
                f'Argon\n{len(x)}\n{atoms}{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n')
            options = (f'sigma={SIGMA * NM[unit]:.12g} epsilon={EPSILON} cutoff={CUTOFF * NM[unit]:.12g} '
                       f'atomic_type=18 length_unit={unit} energy_unit=\"kJ/mol\"')
            # the classical cutoffs bound the domain size; the model does not use them
            cutoff = 0.5 if name in ('dd-thin', 'dd-empty-ranks') else 1.0
            (root / f'{name}.mdp').write_text(f'''integrator = md
nsteps = 0
dt = 0.001
nstlist = 1
cutoff-scheme = Verlet
verlet-buffer-tolerance = -1
rlist = {cutoff + 0.1:g}
rcoulomb = {cutoff:g}
rvdw = {cutoff:g}
coulombtype = Cut-off
vdwtype = Cut-off
pbc = {pbc}
nstenergy = 1
nstcalcenergy = 1
nstfout = 1
nstxout = 1
tcoupl = no
pcoupl = no
constraints = none
comm-mode = none
metatomic-active = true
metatomic-input-group = System
metatomic-model = metatomic-lj-model
metatomic-plugin = {plugin}
metatomic-model-options = {options}
metatomic-check-consistency = true
''' + (walls if pbc == 'xy' else ''))
            run([gmx, 'grompp', '-f', f'{name}.mdp', '-c', f'{name}.gro', '-p', f'{name}.top',
                 '-o', f'{name}.tpr'], cwd=root)
            run(mdrun_command(gmx, ['mdrun', '-s', f'{name}.tpr', '-deffnm', name, '-nb', 'cpu'],
                              dd, launcher, library), cwd=root)
            log = (root / f'{name}.log').read_text()
            assert 'Metatomic model: length unit' in log, 'the metatomic force provider was not initialized'
            if ranks > 1:
                assert 'Domain decomposition grid {} x {} x {}'.format(*dd) in log, f'{name}: no {ranks}-rank DD'
                assert 'each rank runs the model on its home atoms' in log, f'{name}: not the DD path'
            run([gmx, 'energy', '-f', f'{name}.edr', '-o', f'{name}-energy.xvg'], cwd=root,
                input='Metatomic-Potential\n0\n')
            run([gmx, 'traj', '-s', f'{name}.tpr', '-f', f'{name}.trr',
                 '-of', f'{name}-forces.xvg'], cwd=root, input='System\n')
            energy = rows(root / f'{name}-energy.xvg')[0][1]
            forces = rows(root / f'{name}-forces.xvg')[0][1:]
            expected_energy, expected_forces = expected(x, box, [axis in pbc for axis in 'xyz'])
            assert math.isclose(energy, expected_energy, rel_tol=2e-4, abs_tol=1e-5), (name, energy, expected_energy)
            expected_forces = [f for atom in expected_forces for f in atom]
            assert len(forces) == 3 * len(x), forces
            for actual, value in zip(forces, expected_forces):
                assert math.isclose(actual, value, rel_tol=2e-4, abs_tol=1e-4), (name, forces, expected_forces)
            print(f'{name} ({ranks} rank{"s" if ranks > 1 else ""}): plugin, tpr options, energy and forces PASS')
        oniom_coulomb(gmx, plugin, root, launcher, library)


if __name__ == '__main__':
    main()
