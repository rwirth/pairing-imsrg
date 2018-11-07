
import numpy as np

class Basis:
    def __init__(self, nlevels):
        from itertools import product, combinations

        self.nlevels = nlevels
        self.spb = tuple( (lvl, ud) for lvl, ud in product(range(nlevels), 'ud') )
        self.tpb = tuple(combinations(range(len(self.spb)), r=2))

        self.nspstates = len(self.spb)
        self.ntpstates = len(self.tpb)

        self._tpb_lut = { state: (i, +1) for i, state in enumerate(self.tpb) }
        self._tpb_lut.update({ (q,p): (i,-1) for i, (p,q) in enumerate(self.tpb) })
        self._tpb_lut.update({ (p,p): (0,0) for p in range(len(self.spb)) })

        self.vacuum = Reference.vacuum(self)

    def get_tpi(self, s1, s2):
        return self._tpb_lut[s1,s2]



class Reference:
    def __init__(self, basis, gam1, gam2=None):
        self.basis = basis
        self.gam1 = np.asarray(gam1)
        if gam2:
            self.gam2 = np.asarray(gam2)
        else:
            self.gam2 = self._calc_gam2(self.basis, self.gam1)

        self.nparticles = int(round(np.sum(gam1)))

    def is_vacuum(self, tol=1e-6):
        return np.all(np.abs(self.gam1) < tol)

    def gam2_single(self, hole=False):
        return self._calc_gam2(self.basis, self.gam1, hole)

    def classifiers(self, hole_levels=None):
        if hole_levels is not None:
            hole_levels_set = set(hole_levels)
        else:
            hole_levels_set = set( lvl for i, (lvl, ud) in enumerate(self.basis.spb) if self.gam1[i] == 1. )

        particles = tuple( i for i, (lvl, ud) in enumerate(self.basis.spb) if lvl not in hole_levels_set )
        holes = tuple( i for i, (lvl, ud) in enumerate(self.basis.spb) if lvl in hole_levels_set )

        check_hh = lambda p,q: p in holes and q in holes
        check_ph = lambda p,q: (p in holes and q in particles) or (p in particles and q in holes)
        check_pp = lambda p,q: p in particles and q in particles

        return holes, particles, check_hh, check_ph, check_pp

    @staticmethod
    def vacuum(basis):
        return Reference(basis, np.array([ 0. for i in range(basis.nspstates) ]))

    @staticmethod
    def single(basis, fermilevel):
        return Reference(basis, np.array([ 1. if lvl < fermilevel else 0. for lvl, ud in basis.spb ]))

    @staticmethod
    def _calc_gam2(basis, gam1, hole=False):
        if hole:
            gam1 = gam1.copy()
            gam1 = 1. - gam1

        gam2 = np.zeros((basis.ntpstates, basis.ntpstates))
        for I, (s1, s2) in enumerate(basis.tpb):
            for J, (ss1, ss2) in enumerate(basis.tpb):
                if s1 == ss1 and s2 == ss2:
                    gam2[I,J] = gam1[s1]*gam1[s2]
                if s1 == ss2 and s2 == ss1:
                    gam2[I,J] -= gam1[s1]*gam1[s2]
        return gam2


class Operator:
    def __init__(self, o0, o1, o2, ref):
        self.o0 = o0
        self.o1 = o1
        self.o2 = o2
        self.ref = ref
        self.basis = ref.basis

    @staticmethod
    def zero(ref):
        o0 = 0.
        o1 = np.zeros((ref.basis.nspstates,ref.basis.nspstates))
        o2 = np.zeros((ref.basis.ntpstates,ref.basis.ntpstates))

        return Operator(o0, o1, o2, ref)

    def pack(self, symmetry=None):
        if symmetry == 'hermitian':
            packed = [(self.o0,)]
            for i in range(self.o1.shape[0]):
                packed.append(self.o1[i,i:])
            for i in range(self.o2.shape[0]):
                packed.append(self.o2[i,i:])
            return np.concatenate(packed)
        elif symmetry == 'antihermitian':
            packed = []
            for i in range(self.o1.shape[0]-1):
                packed.append(self.o1[i,i+1:])
            for i in range(self.o2.shape[0]-1):
                packed.append(self.o2[i,i+1:])
            return np.concatenate(packed)
        elif symmetry is None or symmetry == 'none':
            return np.concatenate(([self.o0], self.o1.ravel(), self.o2.ravel()))
        else:
            raise ValueError('Unknown symmetry: {}'.format(symmetry))

    @staticmethod
    def unpack(y, ref, symmetry=None):
        if symmetry == 'hermitian':
            n = 0
            o0 = y[0]
            n += 1

            o1 = np.zeros((ref.basis.nspstates,ref.basis.nspstates))
            for i in range(ref.basis.nspstates):
                o1[i,i:] = y[n:n+ref.basis.nspstates-i]
                o1[i:,i] = y[n:n+ref.basis.nspstates-i]
                n += ref.basis.nspstates-i

            o2 = np.zeros((ref.basis.ntpstates,ref.basis.ntpstates))
            for i in range(ref.basis.ntpstates):
                o2[i,i:] = y[n:n+ref.basis.ntpstates-i]
                o2[i:,i] = y[n:n+ref.basis.ntpstates-i]
                n += ref.basis.ntpstates-i
        if symmetry == 'antihermitian':
            n = 0
            o0 = 0

            o1 = np.zeros((ref.basis.nspstates,ref.basis.nspstates))
            for i in range(ref.basis.nspstates-1):
                o1[i,i+1:] = y[n:n+ref.basis.nspstates-i-1]
                o1[i+1:,i] = -y[n:n+ref.basis.nspstates-i-1]
                n += ref.basis.nspstates-i-1

            o2 = np.zeros((ref.basis.ntpstates,ref.basis.ntpstates))
            for i in range(ref.basis.ntpstates-1):
                o2[i,i+1:] = y[n:n+ref.basis.ntpstates-i-1]
                o2[i+1:,i] = -y[n:n+ref.basis.ntpstates-i-1]
                n += ref.basis.ntpstates-i-1
        elif symmetry is None or symmetry == 'none':
            i = 0

            o0 = y[0]
            i += 1

            o1 = y[i:i+ref.basis.nspstates**2].reshape(ref.basis.nspstates,ref.basis.nspstates)
            i += ref.basis.nspstates**2

            o2 = y[i:].reshape(ref.basis.ntpstates, ref.basis.ntpstates)

        return Operator(o0, o1, o2, ref)

    def normalorder(self, newref):
        from itertools import product
        dim2 = self.basis.ntpstates

        if self.ref.is_vacuum():
            v0, v1 = self.o0, self.o1
        else:
            v0 = self.o0 - np.einsum('ii,i', self.o1, self.ref.gam1) - np.einsum('ij,ij', self.o2, self.ref.gam2 - 2 * self.ref.gam2_single())
            v1 = self.o1.copy()
            for (I, (p,r)), (J, (q,s)) in product(enumerate(self.basis.tpb), repeat=2):
                if r == s:
                    v1[p,q] -= self.o2[I,J] * self.ref.gam1[r]
                if r == q:
                    v1[p,s] += self.o2[I,J] * self.ref.gam1[r]
                if p == s:
                    v1[r,q] += self.o2[I,J] * self.ref.gam1[p]
                if p == q:
                    v1[r,s] -= self.o2[I,J] * self.ref.gam1[p]

        if newref.is_vacuum():
            return Operator(v0, v1, self.o2, newref)
        else:
            o0 = v0 + np.einsum('ii,i', v1, newref.gam1) + np.einsum('ij,ij', self.o2, newref.gam2)
            o1 = v1.copy()
            for (I, (p,r)), (J, (q,s)) in product(enumerate(self.basis.tpb), repeat=2):
                if r == s:
                    o1[p,q] += self.o2[I,J] * newref.gam1[r]
                if r == q:
                    o1[p,s] -= self.o2[I,J] * newref.gam1[r]
                if p == s:
                    o1[r,q] -= self.o2[I,J] * newref.gam1[p]
                if p == q:
                    o1[r,s] += self.o2[I,J] * newref.gam1[p]
            return Operator(o0, o1, self.o2, newref)

    def comm(self, other):
        from itertools import product

        if self.ref is not other.ref:
            raise ValueError('Reference states must be identical')

        gam1 = self.ref.gam1
        gam2s = self.ref.gam2_single()
        gambar2s = self.ref.gam2_single(hole=True)
        # lam2 = self.ref.gam2 - gam2s

        one_minus_nn = np.array([ 1 - gam1[t] - gam1[v] for t, v in self.basis.tpb ])

        o1o1_comm = np.dot(self.o1, other.o1) - np.dot(other.o1, self.o1)

        c2 = np.zeros((self.basis.ntpstates,self.basis.ntpstates))

        # Local function for fine-grained profiling
        def _calc_c2():
            nonlocal c2

            # Embed 1B part into two-body space
            o12 = np.zeros((self.basis.ntpstates,self.basis.ntpstates))
            oo12 = np.zeros((self.basis.ntpstates,self.basis.ntpstates))
            for (I, (p,r)), (J, (q,s)) in product(enumerate(self.basis.tpb), repeat=2):
                if r == s:
                    o12[I,J] += self.o1[p,q]
                    oo12[I,J] += other.o1[p,q]
                if p == q:
                    o12[I,J] += self.o1[r,s]
                    oo12[I,J] += other.o1[r,s]
                if p == s:
                    o12[I,J] -= self.o1[r,q]
                    oo12[I,J] -= other.o1[r,q]
                if r == q:
                    o12[I,J] -= self.o1[p,s]
                    oo12[I,J] -= other.o1[p,s]

            c2 += np.dot(self.o2, oo12) - np.dot(oo12, self.o2)
            c2 += np.dot(o12, other.o2) - np.dot(other.o2, o12)

            o2_ph = np.zeros((self.basis.nspstates,)*4)
            oo2_ph = np.zeros((self.basis.nspstates,)*4)
            for (I, (p,r)), (J, (q,s)) in product(enumerate(self.basis.tpb), repeat=2):
                o2_ph[p,s,q,r] = -self.o2[I,J]
                o2_ph[r,s,q,p] = self.o2[I,J]
                o2_ph[p,q,s,r] = self.o2[I,J]
                o2_ph[r,q,s,p] = -self.o2[I,J]

                oo2_ph[p,s,q,r] = -other.o2[I,J]
                oo2_ph[r,s,q,p] = other.o2[I,J]
                oo2_ph[p,q,s,r] = other.o2[I,J]
                oo2_ph[r,q,s,p] = -other.o2[I,J]

            o2_ph.shape = (self.basis.nspstates**2, self.basis.nspstates**2)
            oo2_ph.shape = (self.basis.nspstates**2, self.basis.nspstates**2)

            nt_nv = gam1[:,np.newaxis] - gam1[np.newaxis,:]
            nt_nv.shape = (self.basis.nspstates**2,)

            comm_ph = np.einsum('ij,j,jk', o2_ph, nt_nv, oo2_ph) - np.einsum('ij,j,jk', oo2_ph, nt_nv, o2_ph)
            comm_ph.shape = (self.basis.nspstates,)*4

            for (I, (p,r)), (J, (q,s)) in product(enumerate(self.basis.tpb), repeat=2):
                c2[I,J] += comm_ph[p,s,q,r] - comm_ph[p,q,s,r]

            c2 += np.einsum('ij,j,jk', self.o2, one_minus_nn, other.o2) - np.einsum('ij,j,jk', other.o2, one_minus_nn, self.o2)
        _calc_c2()

        c1_intermediate_gam = np.einsum('ij,jk,kl', self.o2, gam2s, other.o2) - np.einsum('ij,jk,kl', other.o2, gam2s, self.o2)
        c1_intermediate_gambar = np.einsum('ij,jk,kl', self.o2, gambar2s, other.o2) - np.einsum('ij,jk,kl', other.o2, gambar2s, self.o2)

        # c1 multi-ref terms not implemented.
        c1 = o1o1_comm.copy()

        # Local function for fine-grained profiling.
        def _calc_c1():
            nonlocal c1
            for (I, (p,r)), (J, (q,t)) in product(enumerate(self.basis.tpb), repeat=2):
                c1[p,q] += (self.o2[I,J]*other.o1[r,t] - other.o2[I,J]*self.o1[r,t]) * (gam1[r] - gam1[t])
                c1[r,q] -= (self.o2[I,J]*other.o1[p,t] - other.o2[I,J]*self.o1[p,t]) * (gam1[p] - gam1[t])
                c1[p,t] -= (self.o2[I,J]*other.o1[r,q] - other.o2[I,J]*self.o1[r,q]) * (gam1[r] - gam1[q])
                c1[r,t] += (self.o2[I,J]*other.o1[p,q] - other.o2[I,J]*self.o1[p,q]) * (gam1[p] - gam1[q])

                if r == t:
                    c1[p,q] += c1_intermediate_gam[I,J] * (1 - gam1[r]) + c1_intermediate_gambar[I,J] * gam1[r]
                if r == q:
                    c1[p,t] -= c1_intermediate_gam[I,J] * (1 - gam1[r]) + c1_intermediate_gambar[I,J] * gam1[r]
                if p == t:
                    c1[r,q] -= c1_intermediate_gam[I,J] * (1 - gam1[p]) + c1_intermediate_gambar[I,J] * gam1[p]
                if p == q:
                    c1[r,t] += c1_intermediate_gam[I,J] * (1 - gam1[p]) + c1_intermediate_gambar[I,J] * gam1[p]
        _calc_c1()

        # c0 multi-ref terms not implemented but simple: + np.einsum('ij,ji', c2, lam2)
        c0 = np.einsum('ii,i', o1o1_comm, gam1) + np.einsum('ij,ji', c1_intermediate_gambar, gam2s)

        return Operator(c0, c1, c2, self.ref)

    def bch(self, omega, rtol=1e-8):
        res = self.copy()
        term = self.copy()
        fct = 1
        m = 1

        termnorm0 = res.norm()
        termnorm = 1.

        while termnorm > rtol:
            term = omega.comm(term)
            termnorm = (term.norm() / fct) / termnorm0
            res += term / fct

            m += 1
            fct *= m

        return res

    def copy(self):
        return Operator(self.o0, self.o1.copy(), self.o2.copy(), self.ref)

    def norm(self):
        return abs(self.o0) + np.linalg.norm(self.o1) + np.linalg.norm(self.o2)

    def __iadd__(self, other):
        self.o0 += other.o0
        self.o1 += other.o1
        self.o2 += other.o2
        return self
    def __isub__(self, other):
        self.o0 -= other.o0
        self.o1 -= other.o1
        self.o2 -= other.o2
        return self
    def __imul__(self, alpha):
        self.o0 *= alpha
        self.o1 *= alpha
        self.o2 *= alpha
        return self
    def __itruediv__(self, alpha):
        self.o0 /= alpha
        self.o1 /= alpha
        self.o2 /= alpha
        return self

    def __add__(self, other):
        return Operator(self.o0, self.o1, self.o2, self.ref).__iadd__(other)
    def __sub__(self, other):
        return Operator(self.o0, self.o1, self.o2, self.ref).__isub__(other)
    def __mul__(self, alpha):
        return Operator(self.o0, self.o1, self.o2, self.ref).__imul__(alpha)
    def __rmul__(self, alpha):
        return self * alpha
    def __truediv__(self, alpha):
        return Operator(self.o0, self.o1, self.o2, self.ref).__itruediv__(alpha)
