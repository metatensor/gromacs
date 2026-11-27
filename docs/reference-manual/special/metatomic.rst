.. _metatomic:

Metatomic potentials
--------------------

The metatomic interface evaluates `metatomic <https://docs.metatensor.org/metatomic/>`__
models, most of them machine-learned potentials, on the atoms of
:mdp:`metatomic-input-group`. Its options are described in :ref:`mdp-metatomic`.
This section describes the ONIOM scheme, which embeds the group in a classical
system.

ONIOM subtractive scheme
^^^^^^^^^^^^^^^^^^^^^^^^

With :mdp:`metatomic-oniom`, the energy of the system is

.. math::

   E = E_\mathrm{MM}(\mathrm{full}) + E_\mathrm{ML}(\mathrm{embedded})
       - E_\mathrm{MM}(\mathrm{embedded}),

where the embedded system is the input group, :math:`E_\mathrm{ML}` is the
energy of the model and :math:`E_\mathrm{MM}` that of the force field.
:ref:`gmx grompp` subtracts :math:`E_\mathrm{MM}(\mathrm{embedded})` from the
topology: it removes the bonds, angles and dihedrals whose atoms are all in the
group, and excludes the non-bonded pairs of the group. Bonded terms with an atom
outside the group stay in :math:`E_\mathrm{MM}(\mathrm{full})`, and so do the
non-bonded interactions between the group and the other atoms.

The non-bonded kernels treat excluded pairs within :mdp:`rcoulomb` only. The
rest of the Coulomb interaction between the embedded atoms is removed
separately, at any distance: with Ewald and PME electrostatics, the
reciprocal-space interaction :math:`q_i q_j\,\mathrm{erf}(\beta r)/r` of pairs
beyond :mdp:`rcoulomb`; with reaction-field and plain cut-off electrostatics,
the term :math:`q_i q_j (k_\mathrm{rf} r^2 - c_\mathrm{rf})` that the kernels
leave on excluded pairs within it. The classical Coulomb energy of the group is
then zero at all distances, apart from the constant reaction-field self term,
and the group may be wider than :mdp:`rcoulomb`. The correction is reported in
the Coulomb (SR) energy term.

Link atoms
^^^^^^^^^^

When the group ends at a covalent bond, the model sees the atom at the cut with
a missing neighbour. With :mdp:`metatomic-link-atoms`, the model sees a hydrogen
atom in place of the outside atom of each such bond, at
:math:`\mathbf{r}_\mathrm{cap} = \mathbf{r}_\mathrm{e} + d\,\mathbf{u}`, where
:math:`\mathbf{r}_\mathrm{e}` is the atom of the group, :math:`\mathbf{u}` the
unit vector along the bond and :math:`d = 0.1` nm. The bond itself and the
other classical terms of the outside atom stay in the topology. The force on a
cap is spread onto the two atoms of its bond with the chain rule, as for a
virtual site, and the virial gets the term of the fixed distance :math:`d`,
:math:`\frac{1}{2} d\,(\mathbf{u}\cdot\mathbf{F}_\mathrm{cap})\,
\mathbf{u}\otimes\mathbf{u}`.

Sites
^^^^^

:mdp:`metatomic-site-groups` divides the group into sites, which the model
evaluates as separate systems, each with its total charge and spin
multiplicity (:mdp:`metatomic-site-charges`,
:mdp:`metatomic-site-spin-multiplicities`). The classical interactions within
each site are removed, and those between sites are kept.
