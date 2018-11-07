#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


from imsrg.operator import Basis, Reference, Operator
from imsrg import wegner_generator, white_generator, integrate_direct, integrate_magnus, integrate_magnus_dopri
from imsrg.ci import SDBasis, SDMatrix


def show_twobody_part(op):
    # Get two-body state classifiers and build a permutation that sorts states in the order hh, ph, pp
    hh_classifier, ph_classifier, pp_classifier = op.ref.classifiers()[2:5]
    perm = sorted(list(range(op.basis.ntpstates)), key=lambda i: 0 if hh_classifier(*op.basis.tpb[i]) else (
        1 if ph_classifier(*op.basis.tpb[i]) else 2))

    # get first ph and pp state index
    first_ph = next(filter(lambda i: ph_classifier(*ref.basis.tpb[perm[i]]), range(ref.basis.ntpstates)), None)
    first_pp = next(filter(lambda i: pp_classifier(*ref.basis.tpb[perm[i]]), range(ref.basis.ntpstates)), None)

    plt.matshow(np.abs(op.o2[np.ix_(perm, perm)]))
    plt.axhline(first_ph-0.5)
    plt.axvline(first_ph-0.5)
    plt.axhline(first_pp-0.5)
    plt.axvline(first_pp-0.5)
    return plt.colorbar()


def pairing_hamiltonian(basis, g, *, energies=None, delta_e=None):
    from itertools import product

    if energies is None and delta_e is None:
        raise ValueError('Either energies or delta_e must be given.')
    if energies is not None and delta_e is not None:
        raise ValueError('Only one of energies and delta_e can be given.')

    if energies is not None and len(energies != basis.nlevels):
        raise ValueError('The energy array must have length = basis.nlevels')

    if delta_e is not None:
        energies = [ i*delta_e for i in range(basis.nlevels) ]

    h0 = 0.

    h1 = np.zeros((basis.nspstates, basis.nspstates))
    for i, (lvl, ud) in enumerate(basis.spb):
        h1[i,i] = energies[lvl]

    h2 = np.zeros((basis.ntpstates, basis.ntpstates))
    for (I, (p,r)), (J, (q,s)) in product(enumerate(basis.tpb), repeat=2):
        plvl, pud = basis.spb[p]
        rlvl, rud = basis.spb[r]
        qlvl, qud = basis.spb[q]
        slvl, sud = basis.spb[s]

        if plvl == rlvl and pud != rud and qlvl == slvl and qud != sud:
            h2[I,J] = -g/2

    return Operator(h0, h1, h2, basis.vacuum)


basis = Basis(4)
fermilevel = 2
g = 0.5
sMax = 15

ref = Reference.single(basis, fermilevel)
sdbasis = SDBasis(basis, ref.nparticles)

hvac = pairing_hamiltonian(basis, g, delta_e=1)
href0 = hvac.normalorder(ref)

print('Starting integration for Pairing Hamiltonian (g={}) with {} levels total and {} filled.'.format(g, basis.nlevels, fermilevel))

def step_monitor(href, stats):
    print('s = {:.6f}, e = {:.6f}, E(2) = {:.6f}'.format(stats.s[-1], stats.E0[-1], stats.E2[-1]))

# Direct integration of the ODE for H, using vode in Adams mode.
#success, href, stats = integrate_direct(href0, sMax, white_generator, step_monitor)

# Magnus integration using vode in Adams mode.
success, omega, href, stats = integrate_magnus(href0, sMax, white_generator, step_monitor)

# Magnus integration using 8th order Dormand-Prince with a maximum step size of 0.1 to check numerics.
#success, omega, href, stats = integrate_magnus_dopri(href0, sMax, white_generator, step_monitor)

print('Integration finished!')

print()

print('Initial Many-Body Hamiltonian')
sdmat = SDMatrix(sdbasis, hvac)
print(sdmat.cfmat)
print('Eigenvalues:', sdmat.eigenvalues())

print('Final Many-Body Hamiltonian')
sdmat = SDMatrix(sdbasis, href.normalorder(basis.vacuum))
print(sdmat.cfmat)
print('Eigenvalues:', sdmat.eigenvalues())

cb = show_twobody_part(href0)
plt.xlabel('$i$')
plt.ylabel('$j$')
cb.set_label('$|V_{ij}|$')
plt.title('Initial two-body part')

cb = show_twobody_part(href)
plt.xlabel('$i$')
plt.ylabel('$j$')
cb.set_label('$|V_{ij}|$')
plt.title('Final two-body part')

plt.figure()
plt.plot(stats.s, stats.E0, label='$E_0$')
plt.plot(stats.s, stats.E2, label=r'$E_0 + \mathrm{MBPT}(2)$')
plt.title('Zero-body flow')
plt.xlabel('$s$')
plt.ylabel('$E$ [MeV]')
plt.legend()

plt.figure()
plt.plot(stats.s, stats.fd)
plt.xlabel('$s$')
plt.ylabel('$E$ [MeV]')
plt.title('Flow dependence of many-body eigenvalues')

try:
    plt.show()
except KeyboardInterrupt:
    pass
