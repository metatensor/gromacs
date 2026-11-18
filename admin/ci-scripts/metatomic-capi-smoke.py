#!/usr/bin/env python3
"""Run a two-atom LJ plugin through grompp/mdrun and check energy and forces.

Usage: python3 metatomic-capi-smoke.py /absolute/path/to/gmx /absolute/path/to/plugin

Serial runs check the model in nm and in Angstrom. Two-rank runs with domain
decomposition (each atom on its own rank, so the pair crosses ranks) check a 3D
periodic box and an xy-periodic slab whose pair is farther apart in z than half
the box height, which must not be shifted along z. Four-rank runs check boxes of
64 and 1000 atoms, whose forces are exchanged densely and as indexed tuples.
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

SIGMA, EPSILON, CUTOFF = 0.34, 0.99, 0.9  # the test plugin's Lennard-Jones, nm and kJ/mol

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
        energy += 4 * EPSILON * (sr6 * sr6 - sr6)
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
            options = f'length_unit={unit} energy_unit=\"kJ/mol\"'
            (root / f'{name}.mdp').write_text(f'''integrator = md
nsteps = 0
dt = 0.001
nstlist = 1
cutoff-scheme = Verlet
verlet-buffer-tolerance = -1
rlist = 1.1
rcoulomb = 1.0
rvdw = 1.0
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
metatomic-model = gromacs-test-pair-model
metatomic-plugin = {plugin}
metatomic-model-options = {options}
metatomic-check-consistency = true
''' + (walls if pbc == 'xy' else ''))
            run([gmx, 'grompp', '-f', f'{name}.mdp', '-c', f'{name}.gro', '-p', f'{name}.top',
                 '-o', f'{name}.tpr'], cwd=root)
            mdrun = ['mdrun', '-s', f'{name}.tpr', '-deffnm', name, '-nb', 'cpu']
            if ranks > 1:
                mdrun += ['-dd', *map(str, dd), '-npme', '0']
            if launcher:
                command = [*launcher, str(ranks), gmx, *mdrun]
            elif library == 'thread_mpi':
                command = [gmx, *mdrun, '-ntmpi', str(ranks)]
            else:
                command = [gmx, *mdrun]
            run(command, cwd=root)
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


if __name__ == '__main__':
    main()
