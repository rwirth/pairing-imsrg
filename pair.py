#!/usr/bin/env python3

import numpy as np

from imsrg.operator import Basis, Reference, Operator
from imsrg import wegner_generator, white_generator, integrate_direct, integrate_magnus, integrate_magnus_dopri
from imsrg.ci import SDBasis, SDMatrix

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
fermilevel = 1
sMax = 15

ref = Reference.single(basis, fermilevel)

hvac = pairing_hamiltonian(basis, g=0.5, delta_e=1)
href = hvac.normalorder(ref)

hh_classifier, ph_classifier, pp_classifier = ref.classifiers()[2:5]
perm = sorted(list(range(basis.ntpstates)), key=lambda i: 0 if hh_classifier(*basis.tpb[i]) else (1 if ph_classifier(*basis.tpb[i]) else 2))

first_ph = next(filter(lambda i: ph_classifier(*basis.tpb[perm[i]]), range(basis.ntpstates)), None)
first_pp = next(filter(lambda i: pp_classifier(*basis.tpb[perm[i]]), range(basis.ntpstates)), None)

import matplotlib.pyplot as plt

#plt.matshow(np.abs(href.o2[np.ix_(perm, perm)]))
#plt.axhline(first_ph-0.5)
#plt.axvline(first_ph-0.5)
#plt.axhline(first_pp-0.5)
#plt.axvline(first_pp-0.5)
#plt.colorbar()
#plt.show()

sdbasis = SDBasis(basis, 2*fermilevel)
sdmat = SDMatrix(sdbasis, hvac)

print(len(sdbasis.cfs))
print(sdmat.cfmat)
print(sdmat.eigenvalues())

#y = href.pack(symmetry='hermitian')
#print(y)
#htest = Operator.unpack(y, ref, symmetry='hermitian')
#print(htest.o0)
#print(htest.o1)
#print(htest.o2)

#plt.matshow(np.abs(htest.o2[np.ix_(perm, perm)]))
#plt.axhline(first_ph-0.5)
#plt.axvline(first_ph-0.5)
#plt.axhline(first_pp-0.5)
#plt.axvline(first_pp-0.5)
#plt.colorbar()
#plt.show()

def step_monitor(href, stats):
    print('s = {:.6f}, e = {:.6f}, E(2) = {:.6f}'.format(stats.s[-1], stats.E0[-1], stats.E2[-1]))

#success, omega, href, stats = integrate_magnus(href, sMax, white_generator, ref, step_monitor)
success, href, stats = integrate_direct(href, sMax, white_generator, ref, step_monitor)

sdmat = SDMatrix(sdbasis, href.normalorder(basis.vacuum))
print(sdmat.cfmat)


plt.matshow(np.abs(href.o2[np.ix_(perm, perm)]))
plt.axhline(first_ph-0.5)
plt.axvline(first_ph-0.5)
plt.axhline(first_pp-0.5)
plt.axvline(first_pp-0.5)
plt.colorbar()

plt.figure()
plt.plot(stats.s, stats.E0)
plt.plot(stats.s, stats.E2)

plt.figure()
plt.plot(stats.s, stats.fd)
plt.show()
