import math
import warnings

import numpy as np

import astropy.stats
import astropy.visualization

import photutils.aperture

import scipy.ndimage

import matplotlib.pyplot as plt

import photutils.aperture


def sigmaclippedstats(data, sigma=3.0, axis=None):
    """
    Return sigma-clipped statistics of the given data.

    This behaves exactly as:

        astropy.stats.sigma_clipped_stats(
            data, sigma=sigma, axis=axis, cenfunc="median", stdfunc="mad_std"
        )

    except that it converts all ndarrays to float32 before returning them.

    Furthermore, for the common case of clipping a stack of 2D arrays, it does
    so row by row, which is much more efficient in terms of memory use.

    :param data: The data of which to calculate the statistics.
    :param sigma: The number of standard deviations for the upper and lower
        clipping limits. Defaults to 3.0
    :param axis: The axis along with to clip the data. Defaults to None.
    :return: The mean, median, and standard deviation of the data.
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", Warning)

        if axis == 0 and len(data.shape) == 3:

            ny = data.shape[1]
            nx = data.shape[2]

            meanimage = np.full([ny, nx], np.nan, dtype="float32")
            medianimage = np.full([ny, nx], np.nan, dtype="float32")
            sigmaimage = np.full([ny, nx], np.nan, dtype="float32")

            for iy in range(ny):
                meanrow, medianrow, sigmarow = astropy.stats.sigma_clipped_stats(
                    data[:, iy, :],
                    sigma=sigma,
                    axis=0,
                    cenfunc="median",
                    stdfunc="mad_std",
                )

                meanimage[iy, :] = meanrow
                medianimage[iy, :] = medianrow
                sigmaimage[iy, :] = sigmarow

            mean = meanimage
            median = medianimage
            sigma = sigmaimage

        else:

            mean, median, sigma = astropy.stats.sigma_clipped_stats(
                data, sigma=sigma, axis=axis, cenfunc="median", stdfunc="mad_std"
            )

    if isinstance(mean, np.ndarray):
        mean = mean.astype("float32")
    if isinstance(median, np.ndarray):
        median = median.astype("float32")
    if isinstance(sigma, np.ndarray):
        sigma = sigma.astype("float32")

    return mean, median, sigma


def clippedmean(data, sigma=3.0, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        mean, median, sigma = sigmaclippedstats(data, sigma=sigma, axis=axis)
    return mean


def clippedsigma(data, sigma=3.0, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        mean, median, sigma = sigmaclippedstats(data, sigma=sigma, axis=axis)
    return sigma


def clippedmeanandsigma(data, sigma=3.0, axis=None):
    mean, median, sigma = sigmaclippedstats(data, sigma=sigma, axis=axis)
    return mean, sigma


def mean(data, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        return np.nanmean(data, axis=axis)


def median(data, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        return np.nanmean(data, axis=axis)


def medianfilter(data, size):
    return scipy.ndimage.median_filter(data, size)


def uniformfilter(data, size):
    return scipy.ndimage.filters.uniform_filter(data, size=size, mode="nearest")


def show(
    data,
    zrange=False,
    zscale=False,
    contrast=0.25,
    zmin=None,
    zmax=None,
    small=False,
    aperturexy=None,
    apertureradius=[],
    aperturecolor="red",
    aperturealpha=1.0,
    secondaperturexy=None,
    secondapertureradius=[],
    secondaperturecolor="blue",
    secondaperturealpha=1.0,
    cmap="viridis",
    invertcmap=False,
    path=None,
    colorbar=True,
):

    if zmin is not None and zmax is not None:
        interval = astropy.visualization.ManualInterval(zmin, zmax)
    elif zrange:
        interval = astropy.visualization.MinMaxInterval()
    else:
        interval = astropy.visualization.ZScaleInterval(contrast=contrast)
    stretch = astropy.visualization.LinearStretch()
    norm = astropy.visualization.ImageNormalize(
        data, interval=interval, stretch=stretch
    )

    ny = data.shape[0]
    nx = data.shape[1]
    nmax = max(ny, nx)

    if np.max(data.shape) > 1000:
        tickinterval = 100
    else:
        tickinterval = int(math.pow(2, int(math.log2(nmax / 16))))
    ticks = list(
        np.linspace(
            -tickinterval * (nmax // 2 // tickinterval),
            +tickinterval * (nmax // 2 // tickinterval),
            1 + 2 * (nmax // 2 // tickinterval),
        )
    )
    if small:
        plt.figure(figsize=(5, 5 * ny / nx))
    else:
        plt.figure(figsize=(10, 10 * ny / nx))
    plt.imshow(data, origin="lower", norm=norm, cmap=cmap)
    # plt.xticks(ticks, rotation=90)
    # plt.yticks(ticks)
    fraction = 0.03
    pad = 0.02
    if colorbar:
        plt.colorbar(
            fraction=fraction,
            aspect=1 / fraction,
            pad=pad,
            shrink=1 / (1 + fraction + pad),
        )

    if aperturexy is not None:
        aperturexy = np.array(aperturexy)
        if isinstance(apertureradius, (int, float)):
            apertureradius = [apertureradius]
        for apertureradius in apertureradius:
            apertures = photutils.aperture.CircularAperture(
                aperturexy, r=apertureradius
            )
            apertures.plot(color=aperturecolor, lw=1.5, alpha=aperturealpha)

    if secondaperturexy is not None:
        secondaperturexy = np.array(secondaperturexy)
        if isinstance(secondapertureradius, (int, float)):
            secondapertureradius = [secondapertureradius]
        for secondapertureradius in secondapertureradius:
            secondapertures = photutils.aperture.CircularAperture(
                secondaperturexy, r=secondapertureradius
            )
            secondapertures.plot(
                color=secondaperturecolor, lw=1.5, alpha=secondaperturealpha
            )

    if path is not None:
        plt.savefig(path)

    plt.show()


############################################################################


def makesplit(data):

    # We have the choice to use the raw planes (00, 01, 10, and 11), the (I, Q,
    # U) planes, or the (I, q, u) planes. One advantage of (I, Q, U) and (I, q,
    # u) over the raw planes is that stars have the same centers in both. One
    # advantage of (I, q, u) over (I, Q, U) is that we can construct it in such
    # a way that errors are uncorrelated between q and u.

    data00 = data[0::2, 0::2]
    data01 = data[0::2, 1::2]
    data10 = data[1::2, 0::2]
    data11 = data[1::2, 1::2]

    # Use the nominal order 0, 45, 90, and 135 degrees.
    I = 0.5 * (data00 + data01 + data11 + data10)
    Q = data00 - data11
    U = data01 - data10

    return makedataIQU(I, Q, U)


def makedataIQU(dataI, dataQ, dataU):
    return np.array([dataI, dataQ, dataU])


def dataI(data):
    return data[0]


def dataQ(data):
    return data[1]


def dataU(data):
    return data[2]
