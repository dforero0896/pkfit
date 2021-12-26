import numba
import numpy as np
import ultranest
from .cubic_spline import cubic_spline_ypp, cubic_spline_eval_sorted
from scipy.interpolate import interp1d


def get_same_abscissas(x1, x2, y1, y2, n):

    xs = [x1, x2]
    ys = [y1, y2]
    # Get x bounds
    min_x = np.min((x1.min(), x2.min()))
    max_x = np.max((x1.max(), x2.max()))

    new_x = np.logspace(np.log(min_x), np.log(max_x), n)

    interps = [interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')(new_x) for x, y in zip(xs, ys)]

    return new_x, interps[0], interps[1]






@numba.njit
def powspec_model(k, alpha, Sigma_nl, B_nw, nuisance_a, klin, plin, knw, plin_nw):

    exponents = np.arange(-2, 3, 1)

    powspec_lin_nw = np.interp(k, knw, plin_nw)
    powspec_lin = np.interp(k, klin, plin)
    k_exp = np.expand_dims(k, -1)**exponents
    O_factor = powspec_lin / (powspec_lin_nw)# + np.dot(k_exp, nuisance_a))
    O_damp = 1. + (O_factor - 1.) * np.exp(-0.5 * k**2 * Sigma_nl**2)

    powspec_sm = B_nw**2 * powspec_lin_nw + np.dot(k_exp, nuisance_a)

    return powspec_sm * O_damp
@numba.njit
def powspec_smooth(k, B_nw, nuisance_a, klin, plin, knw, plin_nw):

    exponents = np.arange(-2, 3, 1)
    
    powspec_lin_nw = interp_loglog(k, knw, plin_nw)
    powspec_lin = interp_loglog(k, klin, plin)
    k_exp = np.expand_dims(k, -1)**exponents
    
    powspec_sm = B_nw**2 * powspec_lin_nw + np.dot(k_exp, nuisance_a)

    return powspec_sm 

@numba.njit
def interp_loglog(x, xp, yp):
    return np.interp(np.log(x), np.log(xp), yp)
#@numba.njit
#def interp_loglog(x, xp, yp):

#    y = np.empty(x.shape, dtype=np.double)
#    #ypp = cubic_spline_ypp(np.log(xp), yp, xp.shape[0])
#    #cubic_spline_eval_sorted(np.log(xp), yp, ypp, xp.shape[0], np.log(x), y, x.shape[0])
#    ypp = cubic_spline_ypp(xp, yp, xp.shape[0])
#    cubic_spline_eval_sorted(xp, yp, ypp, xp.shape[0], x, y, x.shape[0])

#    return y


@numba.njit
def chisq(kobs, pobs, alpha, Sigma_nl, B_nw, klin, plin, knw, plin_nw, inv_cov):


    exponents = np.arange(-2, 3, 1)

    powspec_lin_nw = interp_loglog(kobs, knw, plin_nw)
    powspec_lin = interp_loglog(kobs, klin, plin)

    k_exp = np.expand_dims(kobs, -1)**exponents
    
    O_factor = interp_loglog(kobs / alpha, klin, plin) / interp_loglog(kobs / alpha, knw, plin_nw)
    O_damp = 1. + (O_factor - 1.) * np.exp(-0.5 * kobs**2 * Sigma_nl**2)
    
    nuisance_term = pobs / O_damp - B_nw**2 * powspec_lin_nw
    design_matrix = k_exp.T.dot(k_exp)
    vector = k_exp.T.dot(nuisance_term)
    nuisance_a = np.linalg.solve(design_matrix, vector)

    
    powspec_sm = B_nw**2 * powspec_lin_nw + np.dot(k_exp, nuisance_a)

    error = pobs - powspec_sm*O_damp 

    return error.T.dot(inv_cov.dot(error))

def get_nuisance(kobs, pobs, alpha, Sigma_nl, B_nw, klin, plin, knw, plin_nw):


    exponents = np.arange(-2, 3, 1)

    powspec_lin_nw = interp_loglog(kobs, knw, plin_nw)
    powspec_lin = interp_loglog(kobs, klin, plin)

    k_exp = np.expand_dims(kobs, -1)**exponents
    
    O_factor = interp_loglog(kobs / alpha, klin, plin) / interp_loglog(kobs / alpha, knw, plin_nw)
    O_damp = 1. + (O_factor - 1.) * np.exp(-0.5 * kobs**2 * Sigma_nl**2)

    nuisance_term = pobs / O_damp - B_nw**2 * powspec_lin_nw
    
    design_matrix = k_exp.T.dot(k_exp)
    vector = k_exp.T.dot(nuisance_term)
    nuisance_a = np.linalg.solve(design_matrix, vector)

    return nuisance_a

@numba.njit
def prior_transform(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    
    lo = 0.8
    hi = 1.2
    params[0] = cube[0] * (hi - lo) + lo
    
    lo = 0.
    hi = 20.
    params[1] = cube[1] * (hi - lo) + lo
    
    lo = 0
    hi = 20.
    params[2] = cube[2] * (hi - lo) + lo
    return params
@numba.njit
def prior_transform_sm(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
       
    lo = 0.
    hi = 20.
    params[0] = cube[0] * (hi - lo) + lo

    return params
def estimate_pk_variance(k, pk, box_size, shot_noise, dk):
    """
    Must use linear k bins.
    Eq. 17 in https://arxiv.org/pdf/2109.15236.pdf
    """
    return (2*np.pi)**3 / box_size**3 * (2 * (pk + shot_noise)**2 / (4*np.pi*k**2 * dk))
def compute_covariance(mocks, kmin, kmax, usecols=(1,), cov_rescale = 1.):
    
    k = np.loadtxt(mocks[0], usecols = (0,))
    k_mask = (k > kmin) & (k < kmax)

    print(f"Computing covariance", flush=True)

    pk = np.squeeze(np.array([np.loadtxt(f, usecols = usecols)[k_mask] for f in mocks]))
    

    N_mocks, N_bins = pk.shape

    error = pk - pk.mean(axis=0)[None, :]

    sample_cov = error.T.dot(error)

    cov_unbiased = cov_rescale * sample_cov / (N_mocks - N_bins - 2)

    inv_cov = np.linalg.inv(cov_unbiased)

    print(f"\tDone", flush=True)
    return cov_unbiased, inv_cov

    

if __name__ == '__main__':
    import time
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import glob
    print(numba.__version__)
    plin_fn = f"/hpcstorage/dforero/projects/baosystematics/data/LinearSpectra/Pk.input_zinit_normalized_at_z0.DAT"
    plin_nw_fn = f"/home/astro/dforero/codes/2pcf_bao_fitter/data/Albert_Pnw.dat"
    mock_dir = f"/hpcstorage/dforero/projects/volume-statistics/data/ft-pk-delta-512"
    mock_list = glob.glob(f"{mock_dir}/CAT*")
    kmin = 0.01; kmax = 0.3


    k = np.linspace(kmin, kmax, 100000)
    klin, plin = np.loadtxt(plin_fn, unpack=True)
    knw, plin_nw = np.loadtxt(plin_nw_fn, unpack=True)

    # kmin = 0.03, kmax = 0.3
    cov, inv_cov = compute_covariance(mock_list, kmin, kmax, usecols=(1,))
    kobs, pobs = np.loadtxt(mock_list[0], usecols=(0,1), unpack=True)
    
    pobs = np.exp(np.interp(np.log(kobs), np.log(klin), np.log(plin)))
    mask_obs = (kobs > kmin) & (kobs < kmax)
    kobs = kobs[mask_obs]
    pobs = pobs[mask_obs]

    @numba.njit
    def log_likelihood_sm(params):
        B_nw = params
        return  -0.5 * chisq(kobs, pobs, 1., 0., B_nw, klin, plin, knw, plin_nw, inv_cov)
    parameters = ['B_nw']
    sampler = ultranest.ReactiveNestedSampler(parameters, log_likelihood_sm, prior_transform_sm, vectorized=False, resume=True, log_dir='test/test_run')
    result = sampler.run(min_num_live_points=1000, min_ess=100) # you can increase these numbers later
    sampler.plot()
    sampler.print_results()
    params = sampler.results['maximum_likelihood']['point']
    nuisance_a = get_nuisance(kobs, pobs, 1., 0., *params, klin, plin, knw, plin_nw)
    pmodel = powspec_smooth(kobs, params[0], nuisance_a, klin, plin, knw, plin_nw)
    
    plt.errorbar(kobs, pobs / interp_loglog(kobs, knw, plin_nw), label='lin', marker='o', markersize=2)
    plt.plot(kobs, pmodel / interp_loglog(kobs, knw, plin_nw), label='sm lin', ls='--')
    plt.legend()
    plt.savefig("test/pkfit.png", dpi=300)

    print(nuisance_a)

    plin_nw = powspec_smooth(knw, params[0], nuisance_a, klin, plin, knw, plin_nw)

    kobs, pobs = np.loadtxt(mock_list[100], usecols=(0,1), unpack=True)
    kobs = kobs[mask_obs]
    pobs = pobs[mask_obs]
    @numba.njit
    def log_likelihood(params):
        alpha, Sigma_nl, B_nw = params
        return  -0.5 * chisq(kobs, pobs, alpha, Sigma_nl, B_nw, klin, plin, knw, plin_nw, inv_cov)
    parameters = ['alpha', 'Sigma_nl', 'B_nw']
    sampler = ultranest.ReactiveNestedSampler(parameters, log_likelihood, prior_transform, vectorized=False, resume=True, log_dir='test/test_run_obs')
    result = sampler.run(min_num_live_points=1000, min_ess=100) # you can increase these numbers later
    sampler.plot()
    sampler.print_results()

    params = sampler.results['maximum_likelihood']['point']
    nuisance_a = get_nuisance(kobs, pobs, *params, klin, plin, knw, plin_nw)
    pmodel = powspec_model(kobs, *params, nuisance_a=nuisance_a, klin=klin, plin=plin, knw=knw, plin_nw=plin_nw)
    psmooth = powspec_smooth(kobs, params[-1], nuisance_a, klin, plin, knw, plin_nw)
    #plt.loglog(knw, plin_nw, label='nw')
    #plt.loglog(klin, plin, label='lin')
    
    plt.errorbar(kobs, pobs / psmooth, yerr = np.sqrt(np.diag(cov))/psmooth, label='data', marker='o', markersize=2, lw=0, elinewidth=2, zorder=10)
    plt.plot(kobs, pmodel / psmooth, label='model', ls='--', zorder=11)
    plt.legend()
    plt.ylabel(rf"$P(k) / P_{{\rm smooth}}(k)$")
    plt.xlabel(rf"$k$")
    plt.savefig("test/pkfit.png", dpi=300)
