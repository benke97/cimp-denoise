"""The 7-number acquisition description CMMP embeds.

    0 pixel size (m)            log10
    1 dwell time (s)            log10
    2 convergence angle (rad)   log10
    3 beam current (A)          log10
    4 HAADF gain                linear
    5 HAADF offset              linear
    6 inner collection angle    linear

z-scored (log10 first on 0-3) with the statistics of CMMP's training set,
exactly as the encoder was trained. The statistics are fixed constants: they
belong to the released CMMP weights.
"""
import numpy as np

COLUMNS = ("pixel size", "dwell time", "convergence angle", "beam current", "gain", "offset",
           "inner collection angle")
LOG_COLS = (0, 1, 2, 3)
PX, DWELL, CONV, CURRENT, GAIN, OFFSET, INNER = range(7)

# stats_mean / stats_std of CMMP_data_16bit_1frame.h5 (the CMMP training set)
S_MEAN = np.array([-10.436434745788574, -5.7213287353515625, -1.7216986417770386, -10.06070613861084,
                   37.39844512939453, -1.7483171224594116, 0.07164888828992844])
S_STD = np.array([0.5755364298820496, 0.3694687485694885, 0.11646884679794312, 0.37764427065849304,
                  6.393189907073975, 0.3136078417301178, 0.013488995842635632])


def z_meta(raw, s_mean=S_MEAN, s_std=S_STD):
    """(..., 7) raw SI values -> z-scored float32. Refuses non-physical input
    (a zero current would otherwise become -inf and silently NaN everything)."""
    r = np.array(raw, np.float64)
    if not np.all(np.isfinite(r)) or np.any(r[..., list(LOG_COLS)] <= 0):
        raise ValueError(f"invalid metadata {r}: pixel size, dwell, convergence and current must be > 0")
    r[..., list(LOG_COLS)] = np.log10(r[..., list(LOG_COLS)])
    return ((r - s_mean) / s_std).astype(np.float32)


def with_dwell(z, dwell_s, s_mean=S_MEAN, s_std=S_STD):
    """Copy of z-scored metadata with the dwell replaced (in seconds)."""
    z = np.array(z, np.float32, copy=True)
    z[..., DWELL] = (np.log10(dwell_s) - s_mean[DWELL]) / s_std[DWELL]
    return z


def dwell_shift(k, s_std=S_STD):
    """z-shift of the dwell column for averaging k frames (dwell x k)."""
    return np.log10(k) / s_std[DWELL]
