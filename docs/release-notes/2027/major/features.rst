New and improved features
^^^^^^^^^^^^^^^^^^^^^^^^^

.. Note to developers!
   Please use """"""" to underline the individual entries for fixed issues in the subfolders,
   otherwise the formatting on the webpage is messed up.
   Also, please use the syntax :issue:`number` to reference issues on GitLab, without
   a space between the colon and number!

Multiple molecule types with identical SETTLE parameters are now supported
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This can be useful when one, for instance, wants to apply position restraints
to a subset of water molecules.

Periodic SCC-DFTB, GFN1-xTB and GFN2-xTB are available for CP2K QM/MM
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The CP2K QM/MM interface can now generate periodic SCC-DFTB, GFN1-xTB and
GFN2-xTB inputs using short-range regularized point-charge coupling with SPME
electrostatics. For GFN1-xTB and GFN2-xTB CP2K should be compiled with support 
for the tblite interface. 
The user can select Gaussian Expansion of Electrostatic Potential (GEEP) 
for MM charges in tight-binding methods through the
``qmmm-cp2k-dftb-electrostatic-coupling = gauss`` MDP option, 
which requires a CP2K 2027.1 or higher, whereas the default is 
short-range regularized point-charge coupling 
``qmmm-cp2k-dftb-electrostatic-coupling = point-charge``.
Regular DFT methods are still using GEEP (Gauss) for electrostatics.

The Metatomic interface runs models through the metatomic-core C API
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The Metatomic force provider no longer links libtorch: models are loaded and
evaluated through the C API of metatomic-core, so any model a metatomic-core
plugin can load works, with options given by :mdp:`metatomic-model-options`.
``-DGMX_METATOMIC=ON`` requires metatomic-core with the API of 545d5976 or
later (pinned to 07cb8766). With domain decomposition every rank evaluates all
model atoms and pairs, which message-passing models need, and the plain
pairlist range is the model's pair-list cutoff plus a buffer, rather than its
interaction range.
