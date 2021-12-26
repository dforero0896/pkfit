"""
Implementation of the "natural" cubic spline interpolation algorithm.
  ref: https://doi.org/10.5281/zenodo.3611922
  see also: https://arxiv.org/abs/2001.09253
"""

import numba
from numba import float32, float64, int64
from numba.pycc import CC
import numpy as np

#cc = CC('cubic_spline')
#cc.verbose = True
INT_MAX = np.iinfo(np.int64).max
INT_MIN = np.iinfo(np.int64).min
OUT_OF_BOUNDS = np.nan


#@cc.export('cubic_spline_ypp', "float64[:](float64[:], float64[:], int64)")
@numba.njit#(["float64[:](float64[:], float64[:], int64)"],
           # fastmath=True)
def cubic_spline_ypp(x, y, n):

    cp = np.empty(n, dtype=np.double)
    ypp = np.empty(n, dtype=np.double)

    newx = x[1]
    newy = y[1]
    c = x[1] - x[0]
    newd = (y[1] - y[0]) / c

    cp[0] = cp[n - 1] = ypp[0] = ypp[n - 1] = 0

    j=1
    while (j < n-1):
        oldx = newx
        oldy = newy
        a = c
        oldd = newd

        newx = x[j+1]
        newy = y[j+1]
        c = newx - oldx
        newd = (newy - oldy) / c

        b = (c + a) * 2
        invd = 1 / (b - a * cp[j - 1])
        d = (newd - oldd) * 6

        ypp[j] = (d - a * ypp[j - 1]) * invd
        cp[j] = c * invd

        j += 1

    while (j):
        j -= 1
        ypp[j] -= cp[j] * ypp[j + 1]
    
    return ypp

#@cc.export('cubic_spline_eval', "float64(float64[:], float64[:], float64[:], float64, int64)")
@numba.njit#(["float64(float64[:], float64[:], float64[:], float64, int64)"],
           # fastmath=True)
def cubic_spline_eval(x, y, ypp, xv, i):

    j = i + 1
    ba = x[j] - x[i]
    xa = xv - x[i]
    bx = x[j] - xv
    ba2 = ba * ba

    lower = xa * y[j] + bx * y[i]
    c = (xa * xa - ba2) * xa * ypp[j]
    d = (bx * bx - ba2) * bx * ypp[i]

    
    return (lower + (1/6) * (c + d)) / ba

#@cc.export('find_abcissa_index', "int64(float64[:], float64, int64, int64)")
@numba.njit#(["int64(float64[:], float64, int64, int64)"],
            #inline='always',
           # fastmath=True)
def find_abcissa_index(x, xv, istart, iend):

    l = istart;
    u = iend;
    while (l <= u):
        i = (l + u) >> 1;
        if (i >= iend):
            if (x[iend] == xv):
                return iend;
            else:
                return INT_MAX;

        if (x[i + 1] <= xv): l = i + 1
        else: 
            if (x[i] > xv): u = i - 1
            else: return i;
    return INT_MAX


#@cc.export('cubic_spline_eval_sorted', "int64(float64[:], float64[:], float64[:], int64, float64[:], float64[:], int64)")
@numba.njit
def cubic_spline_eval_sorted(x, y, ypp, n, xv, yv, nv):

    if xv[0] < x[0]:
        print(f"Out of bounds interp. lower.  (point < limit).", xv[0], "<", x[0])
        assert xv[0] >= x[0]
    if xv[nv-1] > x[n-1]:
        print(f"Out of bounds interp. upper.  (point > limit).", xv[n-1], ">", x[n-1])
        assert xv[nv-1] <= x[n-1]
    end = find_abcissa_index(x, xv[nv - 1], 0, n - 1);
    if (end >= n - 1):
        if (xv[nv - 1] == x[n - 1]): yv[nv - 1] = y[n - 1];
        else:
            raise ValueError("Out of bounds interpolation") 
            
                
            
    
    else: yv[nv - 1] = cubic_spline_eval(x, y, ypp, xv[nv - 1], end);
    if (nv == 1): return 0;

    
    if (end < n - 1): end += 1;
    pos = 0;
    for i in range(nv):
        pos = find_abcissa_index(x, xv[i], pos, end)
        if (pos >= n - 1):
            if (xv[i] == x[n - 1]): yv[i] = y[n - 1];
            else: raise ValueError("Out of bounds interpolation") 
                
        else: yv[i] = cubic_spline_eval(x, y, ypp, xv[i], pos);
    
    return 0;
    

@numba.njit#(["int64(float64[:], float64[:], float64[:], int64, float64[:], float64[:], int64, boolean, float64)"],
            #fastmath=True)
def cubic_spline_eval_sorted_bounds(x, y, ypp, n, xv, yv, nv, bounds_error, fill_value):

    end = find_abcissa_index(x, xv[nv - 1], 0, n - 1);
    if (end >= n - 1):
        if (xv[nv - 1] == x[n - 1]): yv[nv - 1] = y[n - 1];
        else:
            if bounds_error:
                raise ValueError("Out of bounds interpolation") 
            else:
                yv[nv-1] = fill_value
                end = n - 1
            
                
            
    
    else: yv[nv - 1] = cubic_spline_eval(x, y, ypp, xv[nv - 1], end);
    if (nv == 1): return 0;

    
    if (end < n - 1): end += 1;
    pos = 0;
    for i in range(nv):
        print(pos)
        if xv[i] > x[0] and xv[i]<x[n-1]:
            pos = find_abcissa_index(x, xv[i], pos, end);
        else:
            if bounds_error:
                raise ValueError("Out of bounds interpolation (upper)") 
            else:
                yv[i] = fill_value
                #end = i
                continue
        if (pos >= n - 1):
            if (xv[i] == x[n - 1]): yv[i] = y[n - 1];
                
        else: yv[i] = cubic_spline_eval(x, y, ypp, xv[i], pos);
    
    return 0;
    



    

if __name__ == '__main__':

    #cc.compile()
    from scipy.interpolate import interp1d
    import time
    #np.random.seed(42)
    x = np.sort(np.random.random(10) * 2*np.pi)
    y = np.sin(x)
    print(x)
    
    y_new = np.empty(100, dtype=np.double)
    #x_new = np.linspace(0, 2*np.pi, 100)
    x_new = np.linspace(min(x), max(x), 100)

    s = time.time()

    ypp = cubic_spline_ypp(x, y, x.shape[0])
    #cubic_spline_eval_sorted_bounds(x, y, ypp, x.shape[0], x_new, y_new, x_new.shape[0], False, np.nan)
    cubic_spline_eval_sorted(x, y, ypp, x.shape[0], x_new, y_new, x_new.shape[0])

    print(time.time() - s)

    
    s = time.time()
    yf = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=np.nan)

    scpy_y = yf(x_new)
    print(time.time() - s)


    import matplotlib.pyplot as plt
    
    plt.plot(x, y, 'or')
    plt.plot(x_new, np.sin(x_new), ls=':')
    plt.plot(x_new, y_new, label = 'numba')
    plt.plot(x_new, scpy_y, label = 'scipy', ls = '--')
    plt.ylim(-2, 2)
    plt.legend()
    plt.savefig('cubic_spline.png', dpi=150)