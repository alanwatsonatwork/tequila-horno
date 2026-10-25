from datetime import datetime

import horno.image

import math
import numpy as np


def exposuretime(header):
    return float(header["EXPTIME"])


def detectortemperature(header):
    return 0.5 * (float(header["SDTTM"]) + float(header["EDTTM"]))


def datamin(header):
    return 0


def datamax(header):
    return 4095


def offsetxslice(header):
    return slice(4110, 4116)


def offsetyslice(header):
    return slice(0, 2998)


def trimyslice(header):
    return slice(0, 2998)


def trimxslice(header):
    return slice(4, 4110)


def trimyslice(header):
    return slice(0, 2998)


def dorotate(header, data):

    # The detector can be rotated on the sky by a multiple of 90 degrees. This
    # rotation is given by SMTMRO in the header. We rotate to the equivalent of
    # 0 degrees. This has two parts. First, a simple rotation of the images on
    # the sky. Second, if we rotate by 90 or 270 degrees, then the 0 and 90
    # degree polarizers are interchanged, the 45 and 135 degree polarizers
    # are interchanged, and Q and U change sign.

    deltaeta = header["SMTMRO"]

    data = np.rot90(data, -round(deltaeta // 90), axes=(1, 2))

    if deltaeta % 180 != 0:
        data = horno.image.makedataIQU(
            horno.image.dataI(data),
            -horno.image.dataQ(data),
            -horno.image.dataU(data),
        )

    # The astroalign package requires contingious arrays.
    data = np.ascontiguousarray(data)

    return data


def dosplit(header, data):

    data00 = data[0::2, 0::2]
    data01 = data[0::2, 1::2]
    data10 = data[1::2, 0::2]
    data11 = data[1::2, 1::2]

    # Use the nominal order 00 = 0, 01 = 45, 10 = 90, and 11 = 135 degrees.
    I = 0.5 * (data00 + data01 + data11 + data10)
    Q = data00 - data11
    U = data01 - data10

    return horno.image.makedataIQU(I, Q, U)


def flatmax(header):
    return 3000


def gain(header):
    return 2.6


def readnoise(header):
    return 2.7
