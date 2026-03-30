import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.stats import norm


class Theory:

    def __init__(self, J0, r, b, sigma, tau, n_modes, n_rings):
        self.sigma = sigma
        self.J0 = J0
        self.r = r
        self.b = b
        self.tau = tau
        self.n_modes = n_modes
        self.n_rings = n_rings
        self.threshold_ss = None
        self.vels = None
        self.threshold = None
        self.delta_psi = None
        self.lmbda = None
        self.xi = None
        self.lmbda_plus = None
        self.lmbda_minus = None
        self.xi_plus = None
        self.xi_minus = None
        self.delta_lmbda = None
        self.avg_lmbda = None
        self.radial_error = None
        self.tangential_error = None

    def f(self, x, k):
        pass

    def lag(self, v):
        return np.arctan(self.tau * v)

    def steady_state_loss_fun(self, x):
        #return (x + (self.J0 * self.f(np.arccos(x), 0) + self.b) / self.r)**2
        return (x + (self.J0 * self.f(x, 0) + self.b) / self.r)**2

    def steady_state_threshold(self):
        res = minimize(lambda x: self.steady_state_loss_fun(x), 0.25, bounds=[(1e-6,1)], tol=1e-8)
        self.threshold_ss = res.x[0]

    def comp_lambda(self, x, v, k):
        return np.abs(self.f(x, k)) / np.sqrt(1 + (k * self.tau * v)**2)

    def comp_xi(self, x, v, k):
        delta_psi = self.lag(v)
        exp_1j_xi = np.exp(1j * k * delta_psi) * self.f(x, k) * (1 - 1j * k * self.tau * v)
        return np.angle(exp_1j_xi)

        # sgns = np.sign(np.real(self.f(x, k)))
        # if k > 1:
        #     y = -sgns * (np.sin(k * delta_psi) + k * self.tau * v * np.cos(k * delta_psi))
        #     x = sgns * (np.cos(k * delta_psi) - k * self.tau * v * np.sin(k * delta_psi))
        #     return np.arctan2(y, x)
        # else:
        #     return 0

    def comp_threshold(self, v_max, n_vels=1000):
        self.steady_state_threshold()
        vs = np.linspace(0, v_max, n_vels)
        v_span = [0, v_max]
        h0 = self.threshold_ss * np.ones(self.n_rings)
        sol = solve_ivp(lambda v, h: self.dh(v, h), v_span, h0, t_eval=vs, atol=1e-6, max_step=v_max/n_vels, method='RK45')
        threshold = np.array(sol.y)
        self.threshold = np.hstack((threshold[::-1,:0:-1], threshold))
        self.vels = np.hstack((-vs[:0:-1], vs))

    def fourier_transform(self, v_max, n_vels=1000):

        self.comp_threshold(v_max, n_vels=n_vels)
        self.delta_psi = self.lag(self.vels)
        self.lmbda = np.zeros((self.n_rings, 2 * n_vels - 1, self.n_modes))
        self.xi = np.zeros((self.n_rings, 2 * n_vels - 1, self.n_modes))

        for k in range(self.n_modes):
            for r in range(self.n_rings):
                self.lmbda[r,:,k] = self.comp_lambda(self.threshold[r], self.vels, k)
                if k > 0:
                    self.xi[r,:,k] = self.comp_xi(self.threshold[r], self.vels, k)

        vels = self.vels * 1000 * 180 / np.pi

        self.lmbda_plus = self.lmbda[self.n_rings//2:]
        self.lmbda_minus = self.lmbda[self.n_rings//2-1::-1]
        self.xi_plus = self.xi[self.n_rings//2:]
        self.xi_minus = self.xi[self.n_rings//2-1::-1]
        self.delta_lmbda = 0.5 * (self.lmbda[self.n_rings//2:] - self.lmbda[self.n_rings//2-1::-1])
        self.avg_lmbda = 0.5 * (self.lmbda[self.n_rings//2:] + self.lmbda[self.n_rings//2-1::-1])

        return vels, self.threshold, self.lmbda_minus, self.lmbda_plus, self.xi_minus, self.xi_plus, self.delta_psi

    def dh(self, v, h):
        pass

    def comp_readout_params(self, v, k, alpha):
        pass

    def comp_readout_error(self, v_train):
        pass


class ContinuumModel:

    def __init__(self, sigma, n_rings):
        w, p = self.comp_input_weights(sigma, n_rings)
        self.inputs_probs = p
        self.inputs_weights = w

    def comp_input_weights(self, sigma, n_rings):
        n_std = 5
        w = np.linspace(-n_std*sigma, n_std*sigma, n_rings)
        p = norm.pdf(w, loc=0, scale=sigma)
        p /= np.sum(p)
        return w, p

    def get_input_params(self):
        return self.inputs_probs, self.inputs_weights

    def _comp_readout_params(self, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi, v, k, alpha):

        p = 2 * self.inputs_probs[self.n_rings//2:]

        E_lmbda_plus2 = np.sum(p * lmbda_plus**2)
        E_lmbda_minus2 = np.sum(p * lmbda_minus**2)
        E_prod = np.sum(p * lmbda_plus * lmbda_minus * np.exp(1j * (xi_plus + xi_minus)))

        c = E_lmbda_plus2 + E_lmbda_minus2 + 2 * np.abs(E_prod) + alpha
        c *= E_lmbda_plus2 + E_lmbda_minus2 - 2 * np.abs(E_prod) + alpha

        chi = (E_lmbda_plus2 + E_lmbda_minus2 + alpha) * np.exp(1j * k * delta_psi)
        chi -= 2 * E_prod * np.exp(-1j * k * delta_psi)
        chi /= c

        lmbda = np.hstack((lmbda_minus[::-1], lmbda_plus))
        xi = np.hstack((xi_minus[::-1], xi_plus))

        z = chi * lmbda * np.exp(-1j * xi) + np.conj(chi) * lmbda[::-1] * np.exp(1j * xi[::-1])

        rho = np.abs(z)
        phi = np.angle(z)

        return rho, phi

    def _comp_readout_error(self, k, rho, phi, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi):

        W = rho * np.exp(1j * phi)
        W = W[self.n_rings//2:]
        x = np.real(W)[:,np.newaxis]
        y = np.imag(W)[:,np.newaxis]

        delta_psi = delta_psi[np.newaxis,:]
        p = 2 * self.inputs_probs[self.n_rings//2:]
        p = p[:,np.newaxis]

        radial_error = (x * np.cos(xi_plus - k * delta_psi) - y * np.sin(xi_plus - k * delta_psi)) * lmbda_plus
        radial_error += (x * np.cos(xi_minus - k * delta_psi) + y * np.sin(xi_minus - k * delta_psi)) * lmbda_minus

        angular_error = (y * np.cos(xi_plus - k * delta_psi) + x * np.sin(xi_plus - k * delta_psi)) * lmbda_plus
        angular_error += (-y * np.cos(xi_minus - k * delta_psi) + x * np.sin(xi_minus - k * delta_psi)) * lmbda_minus

        radial_error = np.sum(p * radial_error, axis=0)
        angular_error = np.sum(p * angular_error, axis=0)

        return radial_error - 1, angular_error

    def comp_readout_params(self, v, k, alpha):
        v *= np.pi / (1000 * 180)
        idx = np.where(self.vels > v)[0][0]

        lmbda_plus = self.lmbda_plus[:,idx,k]
        lmbda_minus = self.lmbda_minus[:,idx,k]
        xi_plus = self.xi_plus[:,idx,k]
        xi_minus = self.xi_minus[:,idx,k]
        delta_psi = self.delta_psi[idx]

        return self._comp_readout_params(lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi, v, k, alpha)

    def comp_readout_error(self, v, k, alpha):
        v *= np.pi / (1000 * 180)
        idx = np.where(self.vels > v)[0][0]

        lmbda_plus = self.lmbda_plus[:,:,k]
        lmbda_minus = self.lmbda_minus[:,:,k]
        xi_plus = self.xi_plus[:,:,k]
        xi_minus = self.xi_minus[:,:,k]
        delta_psi = self.lag(self.vels)

        rho, phi = self._comp_readout_params(lmbda_plus[:,idx], lmbda_minus[:,idx], xi_plus[:,idx], xi_minus[:,idx], delta_psi[idx], v, k, alpha)
        radial_error, angular_error = self._comp_readout_error(k, rho, phi, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi)

        return self.vels, radial_error, angular_error


class DoubleRingModel:

    def _comp_readout_params(self, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi, v, k, alpha):

        delta_lmbda = 0.5 * (lmbda_plus - lmbda_minus)
        avg_lmbda = 0.5 * (lmbda_plus + lmbda_minus)

        a = lmbda_minus * np.exp(1j * (xi_minus - k * delta_psi))
        b = lmbda_plus * np.exp(-1j * (xi_plus - k * delta_psi))
        z = -(4 * avg_lmbda * delta_lmbda - alpha) * a + (4 * avg_lmbda * delta_lmbda + alpha) * b
        c = (4 * delta_lmbda**2 + alpha) * (4 * avg_lmbda**2 + alpha)
        z /= c

        rho = np.abs(z)
        phi = np.angle(z)

        return rho, phi

    def _comp_readout_error(self, k, rho, phi, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi):

        W = rho * np.exp(1j * phi)
        x = np.real(W)
        y = np.imag(W)

        radial_error = (x * np.cos(xi_plus - k * delta_psi) - y * np.sin(xi_plus - k * delta_psi)) * lmbda_plus
        radial_error += (x * np.cos(xi_minus - k * delta_psi) + y * np.sin(xi_minus - k * delta_psi)) * lmbda_minus

        angular_error = (y * np.cos(xi_plus - k * delta_psi) + x * np.sin(xi_plus - k * delta_psi)) * lmbda_plus
        angular_error += (-y * np.cos(xi_minus - k * delta_psi) + x * np.sin(xi_minus - k * delta_psi)) * lmbda_minus

        return radial_error - 1, angular_error

    def comp_readout_params(self, v, k, alpha):
        v *= np.pi / (1000 * 180)
        idx = np.where(self.vels > v)[0][0]

        lmbda_plus = self.lmbda_plus[0,idx,k]
        lmbda_minus = self.lmbda_minus[0,idx,k]
        xi_plus = self.xi_plus[0,idx,k]
        xi_minus = self.xi_minus[0,idx,k]
        delta_psi = self.delta_psi[idx]

        return self._comp_readout_params(lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi, v, k, alpha)

    def comp_readout_error(self, v, k, alpha):
        v *= np.pi / (1000 * 180)
        idx = np.where(self.vels > v)[0][0]

        lmbda_plus = self.lmbda_plus[0,:,k]
        lmbda_minus = self.lmbda_minus[0,:,k]
        xi_plus = self.xi_plus[0,:,k]
        xi_minus = self.xi_minus[0,:,k]
        delta_psi = self.lag(self.vels)

        rho, phi = self._comp_readout_params(lmbda_plus[idx], lmbda_minus[idx], xi_plus[idx], xi_minus[idx], delta_psi[idx], v, k, alpha)
        radial_error, angular_error = self._comp_readout_error(k, rho, phi, lmbda_plus, lmbda_minus, xi_plus, xi_minus, delta_psi)

        return self.vels, radial_error, angular_error


class ReLUCosTheory(Theory):

    def f0_small(self, x):
        return -self.r * x

    def f0_medium(self, x):
        return self.r * (np.sqrt(1 -x**2) - x * np.arccos(x)) / np.pi

    def f0_large(self, x):
        return 0 * x

    def function_cases(self, x, f_small, f_medium, f_large):
        if hasattr(x, "__len__"):
            out = np.zeros_like(x)
            mask_small = x < -1
            out[mask_small] = f_small(x[mask_small])
            mask_medium = np.logical_and(x >= -1, x < 1)
            out[mask_medium] = f_medium(x[mask_medium])
            mask_large = x > 1
            out[mask_large] = f_large(x[mask_large])
            return out
        else:
            if x < -1:
                return f_small(x)
            elif x > 1:
                return f_large(x)
            else:
                return f_medium(x)

    def f0(self, x):
        return self.function_cases(x, self.f0_small, self.f0_medium, self.f0_large)

    def f1(self, x):
        return self.function_cases(x, self.f1_small, self.f1_medium, self.f1_large)

    def f1_small(self, x):
        return -0.5 * self.r

    def f1_medium(self, x):
        y = np.arccos(x)
        return self.r * (y - np.sin(y) * np.cos(y)) / (2 * np.pi)

    def f1_large(self, x):
        return 0

    def fh(self, x, k):
        return self.function_cases(x, self.fh_small, lambda x: self.fh_medium(x, k), self.fh_large)

    def fh_small(self, x):
        return 0

    def fh_medium(self, x, k):
        y = np.arccos(x)
        return self.r * (k * np.sin(y) * np.cos(k * y) - np.cos(y) * np.sin(k * y)) / ((k - k**3) * np.pi)

    def fh_large(self, x):
        return 0

    def f(self, x, k):
        if k == 0:
            return self.f0(x)
        elif k == 1:
            return self.f1(x)
        else:
            return self.fh(x, k)


class GeneralTheory(Theory):

    def __init__(self, J0, r, b, sigma, tau, n_modes, n_rings, G, dG, h):
        Theory.__init__(self, J0, r, b, sigma, tau, n_modes, n_rings) 
        self.eps = 1e-6  # for numerical differentiation
        self.G = G
        self.dG = dG
        self.h = h
        self.N = 10000
        self.theta = np.linspace(-np.pi, np.pi, self.N, endpoint=False)

    def f0(self, x):
        return np.mean(self.G(self.r * (self.h(self.theta) - x)))

    def f(self, x, k):
        out = self.G(self.r * (self.h(self.theta[np.newaxis,:]) - x[:,np.newaxis]))
        if k == 0:
            return np.mean(out, axis=1)
        else:
            return np.mean(out * np.cos(k * self.theta[np.newaxis,:]), axis=1)

    def df0(self, x):
        if hasattr(x, "__len__"):
            out = -self.r * self.dG(self.r * (self.h(self.theta[np.newaxis,:]) - x[:,np.newaxis]))
            return np.mean(out, axis=1)
        else:
            out = -self.r * self.dG(self.r * (self.h(self.theta) - x))
            return np.mean(out)

        # if hasattr(x, "__len__"):
        #     return (self.f(x + self.eps, 0) - self.f(x, 0)) / self.eps
        # else:
        #     return (self.f0(x + self.eps) - self.f0(x)) / self.eps


class OptimizedGeneralTheory(Theory):

    def __init__(self, J0, r, b, sigma, tau, n_modes, n_rings, G, dG, h):
        Theory.__init__(self, J0, r, b, sigma, tau, n_modes, n_rings) 
        self.eps = 1e-6  # for numerical differentiation
        self.G = G
        self.dG = dG
        self.h = h
        self.N = 10000
        self.theta = np.linspace(-np.pi, np.pi, self.N, endpoint=False)
        self._precompute_f_tables()

    def _precompute_f_tables(self, x_min=-3, x_max=3, n_points=10000):
        """Pre-compute f(x,k) and df0 lookup tables"""
        self._x_grid = np.linspace(x_min, x_max, n_points)
        self._dx = (x_max - x_min) / (n_points - 1)
        self._x_min = x_min
        
        h_theta = self.h(self.theta)
        self._f_tables = {}
        
        x_expanded = self._x_grid[:, np.newaxis]
        h_expanded = h_theta[np.newaxis, :]
        g_vals = self.G(self.r * (h_expanded - x_expanded))
        
        for k in range(self.n_modes):
            if k == 0:
                self._f_tables[k] = np.real(np.mean(g_vals, axis=1))
            elif k == 1:
                weighted_g = g_vals * np.cos(self.theta)[np.newaxis, :]
                self._f_tables[k] = np.mean(weighted_g, axis=1)
            else:
                #cos_k = np.cos(k * self.theta)
                weighted_g = g_vals * np.exp(1j * k * self.theta)[np.newaxis, :]
                self._f_tables[k] = np.mean(weighted_g, axis=1)
        
        # Also pre-compute df0
        dg_vals = -self.r * self.dG(self.r * (h_expanded - x_expanded))
        self._df0_table = np.mean(dg_vals, axis=1)

    def _find_lookup_indices(self, x):
        """Find indices in lookup table for given x values"""
        x = np.asarray(x)
        indices = np.round((x - self._x_min) / self._dx).astype(int)
        indices = np.clip(indices, 0, len(self._x_grid) - 1)
        return indices

    def _interpolate_lookup(self, x, table):
        """
        Linear interpolation in lookup table for smoother results
        """
        x = np.asarray(x)
        
        # Find continuous indices (not rounded)
        continuous_indices = (x - self._x_min) / self._dx
        continuous_indices = np.clip(continuous_indices, 0, len(self._x_grid) - 1)
        
        # Get integer parts and fractional parts
        i0 = np.floor(continuous_indices).astype(int)
        i1 = np.minimum(i0 + 1, len(self._x_grid) - 1)
        frac = continuous_indices - i0
        
        # Linear interpolation: y = y0 + frac * (y1 - y0)
        y0 = table[i0]
        y1 = table[i1]
        result = y0 + frac * (y1 - y0)
        
        return result

    def f0(self, x):
        return self.f(x, 0)

    def f(self, x, k, interpolate=True):
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        if scalar_input:
            x = x[np.newaxis]
        
        if interpolate:
            result = self._interpolate_lookup(x, self._f_tables[k])
        else:
            indices = self._find_lookup_indices(x)
            result = self._f_tables[k][indices]
        
        if scalar_input:
            return result[0]
        return result

    def df0(self, x, interpolate=True):
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        if scalar_input:
            x = x[np.newaxis]
        
        if interpolate:
            result = self._interpolate_lookup(x, self._df0_table)
        else:
            indices = self._find_lookup_indices(x)
            result = self._df0_table[indices]
        
        if scalar_input:
            return result[0]
        return np.real(result)


class DoubleRingGeneralTheory(DoubleRingModel, GeneralTheory):

    def dh(self, v, h):

        c0 = self.df0(h[1])
        c1 = self.df0(h[0])
        c2 = c0 + c1

        dh0 = 2 * self.sigma / self.r * (self.r + self.J0 * c0) / (2 * self.r + self.J0 * c2)
        dh1 = -2 * self.sigma / self.r * (self.r + self.J0 * c1) / (2 * self.r + self.J0 * c2)

        return [dh0, dh1]
        

class DoubleRingReLUCosTheory(DoubleRingModel, ReLUCosTheory):

    def dh(self, v, h):
        if h[1] < -1 and h[0] > 1:
            dh1 = -2 * self.sigma / (self.r * (2 - self.J0))
            dh0 = 2 * self.sigma / self.r  * (1 - self.J0) / (2 - self.J0)
        elif h[1] < -1 and h[0] < 1:
            x1 = np.arccos(h[0])
            dh1 = -2 * self.sigma / self.r * (np.pi - self.J0 * x1) / (2 * np.pi - self.J0 * (np.pi + x1))
            dh0 = 2 * self.sigma / self.r * np.pi * (1 - self.J0) / (2 * np.pi - self.J0 * (np.pi + x1))
        elif h[1] > -1 and h[0] > 1:
            x0 = np.arccos(h[1])
            dh1 = -2 * self.sigma / self.r * np.pi / (2 * np.pi - self.J0 * x0)
            dh0 = 2 * self.sigma / self.r * (np.pi - self.J0 * x0) / (2 * np.pi - self.J0 * x0)
        else:
            x = np.arccos(h)
            dh1 = -2 * self.sigma / self.r * (np.pi - self.J0 * x[0]) / (2 * np.pi - self.J0 * np.sum(x))
            dh0 = 2 * self.sigma / self.r * (np.pi - self.J0 * x[1]) / (2 * np.pi - self.J0 * np.sum(x))
        return [dh0, dh1]


class ContinuumGeneralTheory(ContinuumModel, OptimizedGeneralTheory):

    def __init__(self, J0, r, b, sigma, tau, n_modes, n_rings, G, dG, h):
        OptimizedGeneralTheory.__init__(self, J0, r, b, sigma, tau, n_modes, n_rings, G, dG, h) 
        ContinuumModel.__init__(self, sigma, n_rings) 

    def dh(self, v, h):
        c = np.sum(self.inputs_probs * self.inputs_weights * self.df0(h)) / (self.r + self.J0 * np.sum(self.inputs_probs * self.df0(h)))
        dh = (-self.inputs_weights + self.J0 * c) / self.r
        return dh


class ContinuumReLUCosTheory(ContinuumModel, ReLUCosTheory):

    def __init__(self, J0, r, b, sigma, tau, n_modes, n_rings):
        ReLUCosTheory.__init__(self, J0, r, b, sigma, tau, n_modes, n_rings) 
        ContinuumModel.__init__(self, sigma, n_rings) 

    def dh(self, v, h):
        x = np.arccos(h)
        x[np.isnan(x)] = 0
        c = np.sum(self.inputs_probs * self.inputs_weights * x) / (np.pi - self.J0 * np.sum(self.inputs_probs * x))
        dh = -(self.inputs_weights + self.J0 * c) / self.r
        return dh
