.. _metatomic:

Metatomic machine learning potentials
-------------------------------------

The metatomic module runs machine learning interatomic potentials that follow
the `metatomic <https://docs.metatensor.org/metatomic/latest/>`_ model
interface. A model receives positions, atomic types, the simulation cell and a
neighbor list, and returns the potential energy. Forces and the virial follow
from backpropagating the gradients of that energy, with respect to the
positions and to a strain tensor. Models are trained outside |Gromacs| and
exported as TorchScript files. One exported model therefore runs unchanged
across every code that implements the interface.

The module differs from the :ref:`NNPot interface <nnpot>` in the contract it
holds with the model. NNPot passes a list of input tensors named in the
:ref:`mdp` file, and the model decides what to do with them. A metatomic model
declares its own requirements instead. |Gromacs| queries the model for the
interaction cutoff, for the outputs it can produce and for the quantities it
expects, then builds the system to match. Neighbor list construction and unit
handling stay on the |Gromacs| side, out of the exported model.

Energies are reported in the energy file as a separate ``Metatomic Pot.``
term, so the machine learning contribution stays distinguishable from the
classical terms.

Domain decomposition
^^^^^^^^^^^^^^^^^^^^

Machine learning potentials are many-body. The energy assigned to an atom
depends on every atom within the model cutoff, and for message-passing
architectures on atoms several cutoffs away. A domain-decomposed run must
therefore give each rank a neighbor list that is complete for the atoms that
rank owns. The halo that classical pair interactions need does not suffice.

The module evaluates the model on the home and halo atoms of each rank. It
restricts per-atom energy contributions to home atoms, so summing over ranks
counts each atom once. Two further communication steps complete the per-rank
neighbor list. A backward ghost exchange adds one forward pulse per
decomposition dimension. That pulse supplies the atoms the standard halo omits
whenever the number of pulses falls below the number of cells along a
dimension. A backward pair exchange then runs a ring protocol, discovering
pairs whose partner a neighboring rank owns. The ratio of the model cutoff to
the smallest cell size bounds the number of hops. Periodic shifts are
recomputed under the minimum image convention on receipt.

Forces on halo atoms return to their owning rank after inference. The domain
decomposition machinery does not reduce ``ForceWithVirial``, so the module
communicates these forces itself. Large systems use an indexed gather, small
ones a dense reduction.

Uncertainty and non-conservative forces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model that declares an ``energy_uncertainty`` output is queried for it at
every step. |Gromacs| reports when the per-atom uncertainty exceeds a
threshold. The check catches a trajectory wandering out of the region the model
was trained on, which a machine learning potential otherwise extrapolates into
silently.

Some models train against force and stress targets directly rather than
against energies. Those declare a ``non_conservative_forces`` output, and
optionally ``non_conservative_stress``. Reading them skips the backward pass
entirely, which is cheaper. It also gives up the guarantee that the forces are
the gradient of a potential. Energy is then no longer conserved, and the
dynamics do not sample a well defined ensemble. Use
:mdp:`metatomic-non-conservative` only when that trade-off is intended.

Software prerequisites
^^^^^^^^^^^^^^^^^^^^^^

The module requires a `LibTorch installation
<https://pytorch.org/get-started/locally/>`_ plus the ``metatomic-torch`` and
``metatensor-torch`` C++ libraries. ``GMX_METATOMIC`` selects the interface and
defaults to ``AUTO``, which enables it when those libraries are found and
disables it silently when they are not. ``TORCH`` requires them and fails
configuration otherwise; ``OFF`` skips the search. See
:ref:`installing with metatomic support` for the versions and for fetching the
libraries during configuration.

When the interface is disabled, |Gromacs| still builds a stub. The stub accepts
:ref:`mdp` files that use these options during preprocessing and refuses to run
them, so :ref:`tpr` files can be prepared on machines without LibTorch.

Usage
^^^^^

Simulations are controlled through :ref:`mdp` options, described in full in
the :ref:`mdp options documentation <mdp-metatomic>`. A minimal input
activates the module and names an exported model:

::

    metatomic-active      = true
    metatomic-model       = pet-mad-v1.0.2.pt
    metatomic-input-group = System

With :mdp:`metatomic-input-group` set to ``System``, the model describes every
atom and no classical interactions remain. A smaller index group leaves the
rest of the system on the classical force field. Interactions between the two
regions are then treated classically, as in the mechanical embedding scheme of
:ref:`qmmm` simulations.

The model's declared preference chooses the inference device.
:mdp:`metatomic-device` overrides that preference, and the environment variable
``GMX_METATOMIC_DEVICE`` overrides both at run time without rewriting the
:ref:`tpr` file. Models that ship custom TorchScript operators need the
directory holding them named in :mdp:`metatomic-extensions`.

Exporting a model belongs to the model's own toolchain rather than to
|Gromacs|. The `metatomic documentation
<https://docs.metatensor.org/metatomic/latest/>`_ covers the export path and
lists the models distributed ready to use. Take care that the LibTorch version
linked into |Gromacs| matches the one used to export the model.
