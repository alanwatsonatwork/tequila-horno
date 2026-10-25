import horno.log

import math
import astropy.io.fits
import matplotlib.pyplot as plt
import horno.bake
import horno.image
import numpy as np
import os.path
import astroalign

np.seterr(invalid="ignore")


def alignobjects(
    headerlist,
    datalist,
    referencedata=None,
    noutput=None,
    name="alignobjects",
    logger=None,
    silent=False,
    verbose=False,
    show=True,
):

    logger = horno.log.getlogger(logger)

    if referencedata is None:
        if verbose:
            logger(name, "using first image as reference.")
        referencedata = datalist[0]

    referencedataI = horno.image.dataI(referencedata)

    aligneddatalist = []

    for header, data in zip(headerlist, datalist):

        fitsname = os.path.basename(header["ORIGNAME"])

        if not silent or verbose:
            logger(name, "aligning object %s." % fitsname)

        dataI = horno.image.dataI(data)

        try:
            T, (sourcexy, targetxy) = astroalign.find_transform(
                dataI,
                referencedataI,
                max_control_points=20,
                min_area=20,
            )
        except:
            logger(name, "warning: unable to align %s." % fitsname)
            continue

        ax = T.params[0, 0]
        ay = T.params[1, 0]
        theta = math.atan2(ay, ax)
        k = round(np.degrees(theta) / 90)
        if k != 0:
            logger(name, "warning: %s has an unexpected rotation." % fitsname)
            continue

        dx = round(T.params[0, 2])
        dy = round(T.params[1, 2])
        if verbose:
            logger(name, "shifts are dy = %+d and dx = %+d." % (dy, dx))

        datanz = data.shape[0]
        if noutput is None:
            noutput = 2 * max(data.shape[1], data.shape[2])

        marginy = (noutput - data.shape[1]) // 2
        marginx = (noutput - data.shape[2]) // 2

        y0 = marginy + dy
        x0 = marginx + dx
        y1 = y0 + data.shape[1]
        x1 = x0 + data.shape[2]

        aligneddata = np.full((datanz, noutput, noutput), np.nan)

        aligneddata[:, y0:y1, x0:x1] = data

        if show:

            sourcexy = np.array(sourcexy) + np.array([x0, y0])
            targetxy = np.array(targetxy) + np.array([marginx, marginy])

            horno.image.show(
                horno.image.dataI(aligneddata),
                aperturexy=sourcexy,
                apertureradius=20,
                aperturecolor="cyan",
                secondaperturexy=targetxy,
                secondapertureradius=40,
                secondaperturecolor="violet",
            )
        aligneddatalist.append(aligneddata)

    if not silent or verbose:
        logger(name, "aligned %d of %d images." % (len(aligneddatalist), len(datalist)))
        logger(name, "finished.")

    return aligneddatalist


def coaddobjects(
    aligneddatalist,
    name="coaddobjects",
    logger=None,
    silent=False,
    verbose=False,
    show=True,
):

    logger = horno.log.getlogger(logger)

    if not silent or verbose:
        logger(name, "coadding %d objects." % len(aligneddatalist))

    coaddeddata = horno.image.mean(aligneddatalist, axis=0)
    if show:
        horno.image.show(horno.image.dataI(coaddeddata))
        horno.image.show(horno.image.dataQ(coaddeddata))
        horno.image.show(horno.image.dataU(coaddeddata))

    if not silent or verbose:
        logger(name, "finished.")

    return coaddeddata
