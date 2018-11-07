
import scipy.linalg as scila
import numpy as np

class Configuration:
    def __init__(self, cf):
        self.cf = tuple(cf)

    def __len__(self):
        return len(self.cf)
    def __getitem__(self, key):
        return self.cf[key]
    def __iter__(self):
        return iter(self.cf)

    def max_coincidence(self, other):
        nparticles = len(self)
        if nparticles != len(other):
            raise ValueError('Configurations must have the same number of particles')

        common = []
        d1 = []
        d2 = []
        phase = 1

        for i, k in enumerate(self):
            for j, l in enumerate(other):
                if k == l:
                    p = len(common)
                    common.append(k)
                    phase *= (-1)**(i-p) * (-1)**(j-p)
                    break
            else:
                d1.append(k)
                phase *= (-1)**(nparticles - len(d1) - i)

        for j, l in enumerate(other):
            if l not in common:
                d2.append(l)
                phase *= (-1)**(nparticles - len(d2) - j)

        return common, d1, d2, phase


class SDBasis:
    def __init__(self, basis, nparticles):
        from itertools import combinations

        def levels_to_comb(lvls):
            comb = []
            for l in lvls:
                comb += [ 2*l, 2*l+1 ]
            return comb

        self.cfs = tuple( Configuration(levels_to_comb(lvls)) for lvls in combinations(range(basis.nspstates//2), nparticles//2) )
        self.basis = basis
        self.nparticles = nparticles

class SDMatrix:
    def __init__(self, sdbasis, hvac):
        from itertools import combinations

        dim = len(sdbasis.cfs)
        mat = np.zeros((dim, dim))

        for I, cf1 in enumerate(sdbasis.cfs):
            for J, cf2 in enumerate(sdbasis.cfs):
                common, d1, d2, phase = cf1.max_coincidence(cf2)

                me = 0.

                d = len(d1)
                if d > 2:
                    continue
                elif d == 2:
                    (i,ph_i), (j, ph_j) = sdbasis.basis.get_tpi(*d1), sdbasis.basis.get_tpi(*d2)
                    me += ph_i * ph_j * phase * hvac.o2[i,j]
                elif d == 1:
                    for q in common:
                        (i,ph_i), (j, ph_j) = sdbasis.basis.get_tpi(q, d1[0]), sdbasis.basis.get_tpi(q, d2[0])
                        me += ph_i * ph_j * phase * hvac.o2[i,j]
                    me += hvac.o1[d1[0],d2[0]]
                else: # d == 0
                    for q, qq in combinations(common, 2):
                        i,ph_i = sdbasis.basis.get_tpi(q, qq)
                        me += phase * ph_i**2 * hvac.o2[i,i]
                    for q in common:
                        me += phase * hvac.o1[q,q]
                    me += hvac.o0

                if me != 0.:
                    mat[I,J] = me

        self.cfmat = mat

    def eigenvalues(self):
        return scila.eigvalsh(self.cfmat)
