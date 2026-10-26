#!/usr/bin/env python3
"""Run a two-atom LJ plugin through grompp/mdrun and check energy and forces.

Usage: python3 metatomic-capi-smoke.py /absolute/path/to/gmx /absolute/path/to/plugin
Only the Python standard library and the built GROMACS executable are needed.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def run(gmx, cwd, *args, input=None):
    result = subprocess.run([gmx, *args], cwd=cwd, input=input, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stdout}")
    return result.stdout


def rows(path):
    return [[float(x) for x in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(('#', '@'))]


def main():
    gmx, plugin = (str(Path(p).resolve(strict=True)) for p in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix='gromacs-capi-') as temporary:
        root = Path(temporary)
        (root / 'atoms.gro').write_text(
            'Two argon atoms\n2\n'
            '    1ARG     Ar    1   0.300   0.400   0.500\n'
            '    2ARG     Ar    2   0.700   0.400   0.500\n'
            '   3.00000   3.00000   3.00000\n')
        (root / 'topol.top').write_text('''[ defaults ]
1 2 no 1 1
[ atomtypes ]
Ar 18 39.948 0.0 A 0.34 0.99
[ moleculetype ]
ARG 1
[ atoms ]
1 Ar 1 ARG Ar 1 0.0 39.948
[ system ]
LJ smoke test
[ molecules ]
ARG 2
''')
        for unit in ['nm', 'Angstrom']:
            options = dict(length_unit=unit, energy_unit='kJ/mol')
            (root / 'run.mdp').write_text(f'''integrator = md
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
pbc = xyz
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
metatomic-extensions = {plugin}
metatomic-model-options = {json.dumps(options)}
metatomic-check-consistency = true
''')
            name = unit.lower()
            run(gmx, root, 'grompp', '-f', 'run.mdp', '-c', 'atoms.gro',
                '-p', 'topol.top', '-o', f'{name}.tpr')
            run(gmx, root, 'mdrun', '-s', f'{name}.tpr', '-deffnm', name, '-ntmpi', '1',
                '-nb', 'cpu')
            log = (root / f'{name}.log').read_text()
            assert 'Metatomic C API' in log, 'C API force provider was not initialized'
            run(gmx, root, 'energy', '-f', f'{name}.edr', '-o', f'{name}-energy.xvg',
                input='Potential\n0\n')
            run(gmx, root, 'traj', '-s', f'{name}.tpr', '-f', f'{name}.trr',
                '-of', f'{name}-forces.xvg', input='System\n')
            energy = rows(root / f'{name}-energy.xvg')[0][1]
            forces = rows(root / f'{name}-forces.xvg')[0][1:]
            sr6 = (0.34 / 0.4) ** 6
            expected_energy = 4 * 0.99 * (sr6 * sr6 - sr6)
            derivative = 4 * 0.99 * (-12 * sr6 * sr6 + 6 * sr6) / 0.4
            assert math.isclose(energy, expected_energy, rel_tol=2e-4, abs_tol=1e-5), (unit, energy, expected_energy)
            expected_forces = [derivative, 0, 0, -derivative, 0, 0]
            assert len(forces) == 6, forces
            for actual, expected in zip(forces, expected_forces):
                assert math.isclose(actual, expected, rel_tol=2e-4, abs_tol=1e-5), (unit, forces, expected_forces)
            print(f'{unit}: plugin loading, tpr options, energy and forces PASS')


if __name__ == '__main__':
    main()
