.. _metatomic:

Metatomic potentials
====================

The Metatomic interface evaluates an exported atomistic model through the
applied-forces framework. The selected atoms are provided to the model as a
Metatomic ``System`` with positions, cell, periodic boundary conditions, atom
types, and engine-built neighbor lists.

For embedded subsystem simulations, |Gromacs| applies the embedded-system
topology preprocessing used by the ONIOM-style subtractive scheme: bonded and
non-bonded molecular-mechanics terms that are represented by the model are
removed from the classical force field contribution, while the model energy is
added through the Metatomic force provider.

Boundary link atoms
-------------------

The :mdp:`metatomic-link-atoms` option enables hydrogen caps at bonds that cross
the Metatomic/MM boundary. Direct MM neighbors bonded to selected Metatomic
atoms are included in the embedded atom set so the bonded terms at the boundary
are handled consistently. During model evaluation, these boundary atoms are
represented as hydrogen link atoms placed along the corresponding boundary
bond. The link-atom position is part of the automatic differentiation graph, so
forces are propagated back to the real atoms connected by the boundary bond.

This option requires a model trained for link-atom-capped boundary chemistry.
If the same boundary MM atom is bonded to multiple selected Metatomic atoms,
each cut bond is represented by a distinct hydrogen cap in the model input.

Model-requested charges
-----------------------

Metatomic models can declare additional inputs through their
``requested_inputs`` metadata. When a model requests ``charges`` or a
``charges/<variant>`` input, |Gromacs| provides a per-atom Metatomic
``TensorMap`` with ``quantity = charge`` and ``unit = e``. Charge inputs are
model-controlled; no :ref:`mdp` option enables or disables them.
