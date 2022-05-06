import numpy as np
from scipy.optimize import least_squares, curve_fit

rng = np.random.default_rng()

x0, A, gamma = 12., 3., 5.

n = 200
x = np.linspace(1., 20., n)
y_exact = A * gamma ** 2. / (gamma ** 2. + (x - x0) ** 2.)

# Add some noise with a sigma of 0.5 apart from a particularly noisy region near x0 where sigma is 3
sigma = np.ones(n) * 0.5
sigma[np.abs(x - x0 + 1.) < 1.] = 3.
noise = rng.normal(size=n) * sigma
y = y_exact + noise


def func(x, x0, A, gamma):
    print("Evaluating at ", (x0, A, gamma))
    """ The Lorentzian entered at x0 with amplitude A and HWHM gamma. """
    return A * gamma ** 2. / (gamma ** 2. + (x - x0) ** 2.)


def residuals(beta):
    return (func(x, *beta) - y) / sigma


def run(method):
    p0 = np.array((10, 4, 2), dtype='float')
    if method == 'curve_fit':
        popt, pcov = curve_fit(func, x, y, p0, sigma=sigma, absolute_sigma=True, epsfcn=1e-7 ** 2.)
    elif method == 'least_squares':
        results = least_squares(residuals, x0=p0, diff_step=1e-7)
        popt, pcov = results.x, np.linalg.inv(results.jac.T @ results.jac)
    else:
        raise NotImplementedError

    yfit = func(x, *popt)
    rms_error = np.linalg.norm(yfit - y_exact)
    print('method : ', method)
    print('Fit parameters:', popt)
    print('sigmas:', pcov.diagonal() ** 0.5)
    print('rms error in fit:', rms_error)
    print()
    return popt, pcov, rms_error


if __name__ == '__main__':
    popt_cf, pcov_cf, rms_cf = run(method='curve_fit')
    popt_ls, pcov_ls, rms_ls = run(method='least_squares')

    print((popt_ls - popt_cf) / popt_cf)
    print(pcov_cf / pcov_ls)
    print(rms_cf / rms_ls)
