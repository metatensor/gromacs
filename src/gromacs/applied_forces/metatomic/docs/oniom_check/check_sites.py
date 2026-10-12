"""End-to-end check of several ML sites (metatomic-site-groups) through the metatomic C API,
serially and with domain decomposition.

Four copies of the 7-atom chain of check_oniom.py are the system; each site is the ML atoms
of one copy (atoms 2-4 by default, with four hydrogen link caps), and copies in no site are
MM only. Copy 0 is wrapped across the periodic boundary,
copy 1 straddles a domain boundary, and copies 2 and 3 are within the model cutoff of each
other, so the model must not see their cross-site pairs. Each site has its own total charge
and spin multiplicity, which the pair model (pair_model.cpp, kappa != 0) turns into a factor
on all the pair energies of that site.

For every rank count, `gmx mdrun -rerun` is checked for

1. the Metatomic energy against an independent evaluation of the model, site by site;
2. the total forces against central finite differences of the energy;
3. the virial against 0.5 dE/d(strain), by central finite differences on sheared boxes;
4. energy, forces and virial against the first rank count (normally serial);
5. the logged assignment of sites to ranks, word for word (with more than one rank).

usage:
    python check_sites.py --gmx path/to/gmx --plugin path/to/libpair_model.so
                          [--ranks 1 2 3 4 2x2x1] [--sites "0 1 2 3"] [--system-energy]

`--sites` lists the sites as `copy` or `copy:atoms`, the atoms as 0-based digits within the
chain (default 123, i.e. atoms 2-4, giving 7 model atoms; "2:12" gives 5, with three caps).
Sites must lie within one molecule. "0 1" leaves copies 2 and 3 to MM, and leaves ranks idle
on more than two ranks. `--ranks` takes a rank count (n x 1 x 1 decomposition) or an explicit
grid such as 2x2x1.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from check_oniom import BASE, D_LINK, DELTA, H, MDP as CHAIN_MDP, NAMES, STRAINS, Z, minimum_image, xvg

HERE = Path(__file__).resolve().parent
BOX = np.diag([6.0, 3.0, 3.0])
UNWRAPPED = BASE + np.where(BASE[:, :1] < 1.0, [3.0, 0.0, 0.0], 0.0)  # x 2.85 .. 3.50
OFFSETS = np.array([[3.0, 0.0, 0.0], [4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.35, 0.0]])
COPIES, PER_COPY = len(OFFSETS), len(BASE)
BONDS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (3, 6)]  # chain.top, 0-based, in order
RC, A, KAPPA = 0.45, 20.0, 0.2
CHARGES, SPINS = [-1, 0, 1, 2], [1, 2, 3, 4]
MDP = CHAIN_MDP.replace("metatomic-link-atoms    = {oniom}", "metatomic-link-atoms    = {oniom}\n{sites}")


def positions():
    jitter = np.random.default_rng(7).uniform(-0.01, 0.01, (COPIES, PER_COPY, 3))
    x = np.concatenate([UNWRAPPED + OFFSETS[c] + jitter[c] for c in range(COPIES)])
    return x - np.floor(x / np.diag(BOX)) * np.diag(BOX)


def links(ml):
    """(embedded, mm) for every bond that ml cuts, in topology bond order."""
    return [(i, j) if i in ml else (j, i) for i, j in BONDS if (i in ml) != (j in ml)]


def model_rows(site):
    """Model atoms of a site (copy, ML atoms): its atoms, then one row per link cap."""
    copy, ml = site
    first = copy * PER_COPY
    return [("atom", first + i) for i in ml] + [("cap", first + e, first + m) for e, m in links(ml)]


def site_energy(x, box, sites, scale):
    pos = []
    for site in sites:
        for row in model_rows(site):
            if row[0] == "atom":
                pos.append((x[row[1]], Z[row[1] % PER_COPY]))
            else:
                bond = minimum_image(x[row[2]] - x[row[1]], box)
                pos.append((x[row[1]] + D_LINK * bond / np.linalg.norm(bond), 1))
    energy = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            r2 = np.sum(minimum_image(pos[j][0] - pos[i][0], box) ** 2)
            if r2 < RC**2:
                energy += scale * A * (1 + 0.1 * (pos[i][1] + pos[j][1])) * (RC**2 - r2) ** 2 / RC**4
    return energy


def site_sizes(sites):
    # a cap takes over its MM atom's row, or gets a row of its own when that atom has one
    return [len(model_rows(site)) for site in sites]


def expected_assignment(sizes, ranks):
    """The same balancing as assignSitesToRanks: largest first, to the least-loaded rank."""
    load, out = [0] * ranks, [[] for _ in range(ranks)]
    for site in sorted(range(len(sizes)), key=lambda s: -sizes[s]):
        rank = load.index(min(load))
        out[rank].append(site)
        load[rank] += sizes[site]
    parts = [f"rank {r}: none" if not s else
             f"rank {r}: {' '.join(f'SITE{k + 1}' for k in sorted(s))} ({sum(sizes[k] for k in s)} model atoms)"
             for r, s in enumerate(out)]
    return (f"Metatomic: {len(sizes)} ML sites on {ranks} ranks, each evaluated whole by one rank; "
            + "; ".join(parts))


def gro(x, box, title):
    lines = [title, str(len(x))]
    for i, r in enumerate(x):
        lines.append(f"{i // PER_COPY + 1:5d}{'MOL':<5s}{NAMES[i % PER_COPY]:>5s}{i + 1:5d}"
                     f"{r[0]:12.7f}{r[1]:12.7f}{r[2]:12.7f}")
    b = box
    lines.append(" ".join(f"{v:.9f}" for v in [b[0, 0], b[1, 1], b[2, 2], b[0, 1], b[0, 2], b[1, 0], b[1, 2], b[2, 0], b[2, 1]]))
    return "\n".join(lines) + "\n"


def index_file(sites):
    groups = {"System": range(COPIES * PER_COPY),
              "ML": [c * PER_COPY + i for c, ml in sites for i in ml]}
    for s, (c, ml) in enumerate(sites):
        groups[f"SITE{s + 1}"] = [c * PER_COPY + i for i in ml]
    return "".join(f"[ {name} ]\n{' '.join(str(a + 1) for a in atoms)}\n" for name, atoms in groups.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gmx", required=True)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--ranks", nargs="+", default=["1", "2", "3", "4", "2x2x1"])
    parser.add_argument("--sites", default="0 1 2 3")
    parser.add_argument("--system-energy", action="store_true")
    args = parser.parse_args()

    sites = [(int(s.split(":")[0]), [int(a) for a in (s.split(":") + ["123"])[1]]) for s in args.sites.split()]
    assert len({c for c, _ in sites}) == len(sites), "one site per copy"
    charges, spins = CHARGES[: len(sites)], SPINS[: len(sites)]
    scales = [1 + KAPPA * (q + 0.1 * m) for q, m in zip(charges, spins)]
    work = Path(tempfile.mkdtemp(prefix="sites_check_"))

    def run(cmd, cwd=work, **kw):
        out = subprocess.run([args.gmx, *map(str, cmd)], cwd=cwd, capture_output=True, text=True, **kw)
        if out.returncode != 0:
            sys.exit(f"gmx {' '.join(map(str, cmd))} failed in {cwd}:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}")
        return out

    (work / "model.pair").write_text(f"{RC} {A} 0 {0 if args.system_energy else 1} {KAPPA}\n")
    site_mdp = (f"metatomic-site-groups   = {' '.join(f'SITE{s + 1}' for s in range(len(sites)))}\n"
                f"metatomic-site-charges  = {' '.join(map(str, charges))}\n"
                f"metatomic-site-spin-multiplicities = {' '.join(map(str, spins))}")
    (work / "rerun.mdp").write_text(MDP.format(model=work / "model.pair", plugin=args.plugin.resolve(),
                                               oniom="yes", sites=site_mdp))
    (work / "topol.top").write_text(re.sub(r"chain 1\s*$", f"chain {COPIES}\n", (HERE / "chain.top").read_text()))
    (work / "index.ndx").write_text(index_file(sites))

    x0 = positions()
    frames = [("base", x0, BOX)]
    for atom in range(len(x0)):
        for dim in range(3):
            for sign in (+1, -1):
                x = x0.copy()
                x[atom, dim] += sign * H
                frames.append((f"x{atom}_{dim}{sign:+d}", x, BOX))
    for a, b in STRAINS:
        for sign in (+1, -1):
            eps = np.eye(3)
            eps[a, b] += sign * DELTA
            frames.append((f"e{a}{b}{sign:+d}", x0 @ eps, BOX @ eps))
    (work / "base.gro").write_text(gro(x0, BOX, "base"))
    (work / "frames.gro").write_text("".join(gro(x, b, t) for t, x, b in frames))
    run(["grompp", "-f", "rerun.mdp", "-c", "base.gro", "-p", "topol.top", "-n", "index.ndx",
         "-o", "topol.tpr", "-maxwarn", "4"])

    reference = sum(site_energy(x0, BOX, [site], scale) for site, scale in zip(sites, scales))
    whole = site_energy(x0, BOX, sites, 1.0)
    print(f"{len(sites)} sites {args.sites!r} (charges {charges}, multiplicities {spins}), work dir {work}")
    print(f"   reference Metatomic energy {reference:.5f} kJ/mol (one system, no site factors: {whole:.5f})")

    ok, first = True, None
    for spec in args.ranks:
        grid = [int(v) for v in spec.split("x")] if "x" in spec else [int(spec), 1, 1]
        n = int(np.prod(grid))
        rundir = work / f"ranks_{spec}"
        rundir.mkdir()
        ranks = ["-ntmpi", n, "-ntomp", 1] + (["-dd", *grid] if n > 1 else [])
        run(["mdrun", "-s", "../topol.tpr", "-rerun", "../frames.gro", *ranks,
             "-o", "rerun.trr", "-e", "rerun.edr", "-g", "rerun.log"], cwd=rundir)
        run(["energy", "-f", "rerun.edr", "-o", "energy.xvg", "-dp"], cwd=rundir,
            input="Potential\nMetatomic-Potential\n\n")
        energies = xvg(rundir / "energy.xvg")
        potential = dict(zip([t for t, _, _ in frames], energies["Potential"]))
        dump = run(["dump", "-f", "rerun.trr"], cwd=rundir).stdout.split("f (")[1].split("frame")[0]
        forces = np.array([[float(v) for v in l.split("{")[1].split("}")[0].split(",")]
                           for l in dump.splitlines() if l.strip().startswith("f[")])
        run(["mdrun", "-s", "../topol.tpr", "-nsteps", 0, *ranks, "-deffnm", "step0"], cwd=rundir)
        terms = [f"Vir-{'XYZ'[a]}{'XYZ'[b]}" for a, b in STRAINS]
        run(["energy", "-f", "step0.edr", "-o", "virial.xvg", "-dp"], cwd=rundir, input="\n".join(terms) + "\n\n")
        virial = np.array([xvg(rundir / "virial.xvg")[name][0] for name in terms])  # gmx energy reorders
        energy = energies["Metatomic Potential"][0]

        print(f"-- {n} rank(s){f' ({spec})' if 'x' in spec else ''}")
        good = abs(energy - reference) < 1e-4 * max(1.0, abs(reference))
        print(f"1. Metatomic energy {energy:.5f} kJ/mol {'ok' if good else 'WRONG'}")
        ok &= good

        fd = np.array([[-(potential[f"x{a}_{d}+1"] - potential[f"x{a}_{d}-1"]) / (2 * H) for d in range(3)]
                       for a in range(len(x0))])
        err = np.abs(fd - forces)
        good = bool(np.all(err < 0.5 + 1e-3 * np.abs(fd)))
        print(f"2. forces: max |GROMACS - finite difference| {err.max():.3f} kJ/mol/nm "
              f"(largest force {np.abs(fd).max():.0f}) {'ok' if good else 'WRONG'}")
        ok &= good

        fd_vir = np.array([0.5 * (potential[f"e{a}{b}+1"] - potential[f"e{a}{b}-1"]) / (2 * DELTA) for a, b in STRAINS])
        good = bool(np.all(np.abs(virial - fd_vir) < 1.0 + 2e-3 * np.abs(fd_vir)))
        print(f"3. virial: max |GROMACS - 0.5 dE/d(strain)| {np.abs(virial - fd_vir).max():.3f} kJ/mol "
              f"{'ok' if good else 'WRONG'}")
        ok &= good

        if first is None:
            first = (spec, energy, forces, virial)
        else:
            de = abs(energy - first[1])
            df = np.abs(forces - first[2]).max()
            dv = np.abs(virial - first[3]).max()
            good = de < 1e-5 * abs(first[1]) + 1e-3 and df < 1e-5 * np.abs(first[2]).max() + 1e-3 and dv < 1e-2
            print(f"4. against {first[0]} rank(s): energy {de:.2e}, forces {df:.2e}, virial {dv:.2e} "
                  f"{'ok' if good else 'WRONG'}")
            ok &= good

        log = (rundir / "rerun.log").read_text()
        if n > 1:
            expected = expected_assignment(site_sizes(sites), n)
            found = [l.strip() for l in log.splitlines() if l.startswith("Metatomic: ") and "ML sites on" in l]
            good = found == [expected]
            print(f"5. log: {found[0] if found else '(no site assignment logged)'} {'ok' if good else 'WRONG'}")
            if not good:
                print(f"   expected: {expected}")
            ok &= good
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
