import numpy as np

from collections import namedtuple
from .operator import Operator
from .ci import SDBasis, SDMatrix

# Copied and modified from https://rosettacode.org/wiki/Bernoulli_numbers
def magnus_coefficients():
    from fractions import Fraction as Fr

    A, m = [], 0
    fct = 1
    while True:
        A.append(Fr(1, m+1))
        for j in range(m, 0, -1):
          A[j-1] = j*(A[j-1] - A[j])
        yield (-1 if m == 1 else +1) * A[0]/fct # (which is Bm)
        m += 1
        fct *= m


def wegner_generator(href, hole_levels=None):
    from numpy import ix_

    holes, particles, hh_classifier, ph_classifier, pp_classifier = href.ref.classifiers(hole_levels)

    hdiag0 = href.o0

    hdiag1 = href.o1.copy()
    hdiag1[ix_(holes, particles)] = 0.
    hdiag1[ix_(particles, holes)] = 0.

    hh_states = tuple( i for i, (p,q) in enumerate(href.basis.tpb) if hh_classifier(p,q) )
    pp_states = tuple( i for i, (p,q) in enumerate(href.basis.tpb) if pp_classifier(p,q) )

    hdiag2 = href.o2.copy()
    hdiag2[ix_(hh_states, pp_states)] = 0.
    hdiag2[ix_(pp_states, hh_states)] = 0.

    hdiag = Operator(hdiag0, hdiag1, hdiag2, href.ref)

    return hdiag.comm(href)

def white_generator(href, hole_levels=None):
    from numpy import ix_

    holes, particles, hh_classifier, ph_classifier, pp_classifier = href.ref.classifiers(hole_levels)

    denom1 = _get_denom1(href, holes, particles)

    gen0 = 0.

    gen1 = np.zeros_like(href.o1)
    gen1[ix_(particles, holes)] = href.o1[ix_(particles, holes)] / denom1
    gen1 -= gen1.T

    denom2, pp_states, hh_states = _get_denom2(href, hh_classifier, ph_classifier, pp_classifier)

    gen2 = np.zeros_like(href.o2)
    gen2[ix_(pp_states, hh_states)] = href.o2[ix_(pp_states, hh_states)] / denom2
    gen2 -= gen2.T

    gen = Operator(gen0, gen1, gen2, href.ref)

    return gen


def mbpt2(href):
    from itertools import product

    hole_levels = set( lvl for i, (lvl, ud) in enumerate(href.basis.spb) if href.ref.gam1[i] > 0. )

    holes, particles, hh_classifier, ph_classifier, pp_classifier = href.ref.classifiers(hole_levels)

    DEpt2 = 0.
    for (I, (p,r)), (J, (q,s)) in product(enumerate(href.basis.tpb), repeat=2):
        if not (hh_classifier(p, r) and pp_classifier(q, s)):
            continue
        DEpt2 -= href.o2[I,J]**2/(href.o1[q,q] + href.o1[s,s] - href.o1[p,p] - href.o1[r,r])

    return href.o0 + DEpt2


def _flow_rhs(generator, reference, hole_levels=None):
    def rhs(s, y):
        href = Operator.unpack(y, reference, symmetry='hermitian')
        eta = generator(href, hole_levels)
        dhds = eta.comm(href)
        return dhds.pack(symmetry='hermitian')
    return rhs


def _magnus_rhs(generator, href0, reference, hole_levels=None, rtol=1e-6):
    def rhs(s, y, href):
        omega = Operator.unpack(y, reference, symmetry='antihermitian')
        href = href0.bch(omega)
        eta = generator(href, hole_levels)

        domegads = eta.copy()
        domegads_term = eta.copy()

        coeffs = magnus_coefficients()
        next(coeffs)

        termnorm0 = domegads.norm()
        termnorm = 1.
        m = 1
        while termnorm > rtol:
            domegads_term = omega.comm(domegads_term)

            coeff = next(coeffs)
            if coeff != 0:
                domegads += float(coeff) * domegads_term
                termnorm = abs(float(coeff)) * domegads_term.norm() / termnorm0
                print('Magnus: order = {}, coeff = {}, ||domega|| = {}, ||rem||_rel = {}'.format(m, coeff, domegads.norm(), termnorm))
            m += 1

        domegads.o0 = 0.
        domegads.o1 = 1/2 * (domegads.o1 - domegads.o1.T)
        domegads.o2 = 1/2 * (domegads.o2 - domegads.o2.T)

        return domegads.pack(symmetry='antihermitian')
    return rhs

def _get_denom1(href, holes, particles):
    denom1 = np.diag(href.o1)[:,np.newaxis] - np.diag(href.o1)[np.newaxis, :]

    for I, (p,q) in enumerate(href.basis.tpb):
        if p in holes and q in particles:
            denom1[q,p] -= href.o2[I,I]
        elif p in particles and q in holes:
            denom1[p,q] -= href.o2[I,I]

    return denom1[np.ix_(particles, holes)]

def _get_denom2(href, hh_classifier, ph_classifier, pp_classifier):
    hh_states = list( i for i, (p,q) in enumerate(href.basis.tpb) if hh_classifier(p,q) )
    pp_states = list( i for i, (p,q) in enumerate(href.basis.tpb) if pp_classifier(p,q) )

    denom = np.array([ href.o1[p,p] + href.o1[q,q] for p,q in href.basis.tpb ])

    denom2 = denom[:,np.newaxis] - denom[np.newaxis,:] + np.diag(href.o2)[:,np.newaxis] + np.diag(href.o2)[np.newaxis,:]

    for I in hh_states:
        h1, h2 = href.basis.tpb[I]
        for J in pp_states:
            p1, p2 = href.basis.tpb[J]

            K1, _ = href.basis.get_tpi(p1, h1)
            K2, _ = href.basis.get_tpi(p2, h2)
            K3, _ = href.basis.get_tpi(p1, h2)
            K4, _ = href.basis.get_tpi(p2, h1)

            denom2[J,I] -= href.o2[K1,K1] + href.o2[K2,K2] + href.o2[K3,K3] + href.o2[K4,K4]

    return denom2[np.ix_(pp_states, hh_states)], pp_states, hh_states


IntegrateStats = namedtuple('IntegrateStats', 's fd E0 E2')

def integrate_direct(href0, s_max, generator, step_monitor=None, convergence_check=None):
    from scipy.integrate import ode

    reference = href0.ref

    solver = ode(_flow_rhs(generator, reference))
    solver.set_integrator('vode')
    solver.set_initial_value(href0.pack(symmetry='hermitian'))

    stats = IntegrateStats([], [], [], [])

    sdbasis = SDBasis(reference.basis, reference.nparticles)

    while solver.t < s_max:
        y = solver.integrate(s_max, step=True)
        href = Operator.unpack(y, reference, symmetry='hermitian')
        Ept2 = mbpt2(href)

        sdmatrix = SDMatrix(sdbasis, href.normalorder(reference.basis.vacuum))

        stats.s.append(solver.t)
        stats.fd.append(sdmatrix.eigenvalues())
        stats.E0.append(href.o0)
        stats.E2.append(Ept2)

        if step_monitor:
            step_monitor(href, stats)

        if not solver.successful():
            break

        if convergence_check is not None and href.o0 - Ept2 < convergence_check:
            break

    return solver.successful(), href, stats


def integrate_magnus(href0, s_max, generator, step_monitor=None, convergence_check=None):
    from scipy.integrate import ode

    reference = href0.ref

    solver = ode(_magnus_rhs(generator, href0, reference))
    solver.set_integrator('vode', atol=1e-6)
    solver.set_initial_value(Operator.zero(reference).pack(symmetry='antihermitian'))
    solver.set_f_params(href0)

    stats = IntegrateStats([], [], [], [])

    sdbasis = SDBasis(reference.basis, reference.nparticles)

    while solver.t < s_max:
        y = solver.integrate(s_max, step=True)
        omega = Operator.unpack(y, reference, symmetry='antihermitian')
        href = href0.bch(omega)
        solver.set_f_params(href)
        Ept2 = mbpt2(href)

        sdmatrix = SDMatrix(sdbasis, href.normalorder(reference.basis.vacuum))

        stats.s.append(solver.t)
        stats.fd.append(sdmatrix.eigenvalues())
        stats.E0.append(href.o0)
        stats.E2.append(Ept2)

        if step_monitor:
            step_monitor(href, stats)

        if not solver.successful():
            break

        if convergence_check is not None and href.o0 - Ept2 < convergence_check:
            break

    return solver.successful(), omega, href, stats


def integrate_magnus_dopri(href0, s_max, generator, step_monitor=None, convergence_check=None):
    from scipy.integrate import ode

    reference = href0.ref

    stats = IntegrateStats([], [], [], [])

    sdbasis = SDBasis(reference.basis, reference.nparticles)

    def solout(s, y):
        nonlocal stats, sdbasis, reference

        omega = Operator.unpack(y, reference, symmetry='antihermitian')
        href = href0.bch(omega)
        Ept2 = mbpt2(href)

        sdmatrix = SDMatrix(sdbasis, href.normalorder(reference.basis.vacuum))

        stats.s.append(s)
        stats.fd.append(sdmatrix.eigenvalues())
        stats.E0.append(href.o0)
        stats.E2.append(Ept2)

        if step_monitor:
            step_monitor(href, stats)

        if convergence_check is not None and href.o0 - Ept2 < convergence_check:
            return -1

    solver = ode(_magnus_rhs(generator, href0, reference))
    solver.set_integrator('dop853', max_step=0.1)
    solver.set_solout(solout)
    solver.set_initial_value(Operator.zero(reference).pack(symmetry='antihermitian'))
    solver.set_f_params(href0)

    y = solver.integrate(s_max)
    omega = Operator.unpack(y, reference, symmetry='antihermitian')
    href = href0.bch(omega)
    return solver.successful(), omega, href, stats
