#THIS IS IN DEVELOPMENT DO NOT SHARE! March 19 2017. stanley.chin@noblis.org
#DoH is not supported

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
import itertools as itt
import math
from math import sqrt, hypot, log
from numpy import arccos
from ..util import img_as_float
from .peak import peak_local_max
from ._hessian_det_appx import _hessian_matrix_det
from ..transform import integral_image
from .._shared.utils import assert_nD


# This basic blob detection algorithm is based on:
# http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
# Theory behind: http://en.wikipedia.org/wiki/Blob_detection (04.04.2013)


def _blob_overlap(blob1, blob2):
    """Finds the overlapping volume fraction between two blobs.

    Returns a float representing fraction of overlapped volume.

    Parameters
    ----------
    blob1 : sequence
        A sequence of ``(z,y,x,sigma)``, where ``x,y,z`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.
    blob2 : sequence
        A sequence of ``(z,y,x,sigma)``, where ``x,y,z`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.

    Returns
    -------
    f : float
        Fraction of overlapped volume.
    """
    root2 = sqrt(2)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[3] * root2
    r2 = blob2[3] * root2
    #peculiar way to compute the distance metric but ok, won't mess with it
    d = hypot(blob1[0] - blob2[0], blob1[1] - blob2[1], blob1[2] - blob2[2])

    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1
    # from http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    volume = math.pi * ((r2 + r1 - d)**2) * (d**2 * 2*d*r1 - 3*r1**2 + 2*d*r2 - 3*r2**2 + 6*r1*r2) / (12 * d)

    return volume / ((4/3)*(math.pi * (min(r1, r2) ** 3)))


def _prune_blobs(blobs_array, overlap):
    """Eliminated blobs with volume overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 3d array with each row representing 4 values, ``(y,x,sigma)``
        where ``(z,y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """

    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1, blob2 in itt.combinations(blobs_array, 2):
        if _blob_overlap(blob1, blob2) > overlap:
            if blob1[3] > blob2[3]:
                blob2[3] = -1
            else:
                blob1[3] = -1

    # return blobs_array[blobs_array[:, 3] > 0]
    return np.array([b for b in blobs_array if b[3] > 0])


def blob_dog3(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
             overlap=.5,):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach

 
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}sigma`.
    """
    #assert_nD(image, 2)

    image = img_as_float(image)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                  * sigma_list[i] for i in range(k)]
    image_cube = np.dstack(dog_images)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  min_distance=3,
                                  threshold_rel=0.0,
                                  exclude_border=False)
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0,3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, 3] = sigma_list[local_maxima[:, 3]]
    local_maxima = lm
    return _prune_blobs(local_maxima, overlap)


def blob_log3(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap = .5, log_scale=False):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

  
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}sigma`.
    """

    #assert_nD(image, 2)

    image = img_as_float(image)

    if log_scale:
        start, stop = log(min_sigma, 10), log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    # s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
    image_cube = np.dstack(gl_images)

    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  min_distance=3),
                                  threshold_rel=0.0,
                                  exclude_border=False)

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0,3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, 3] = sigma_list[local_maxima[:, 3]]
    local_maxima = lm
    return local_maxima

#not supported until i can figure out how to do the slices in 3d space
#def blob_doh3(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01,
             overlap=.5, log_scale=False):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.

    Parameters
    ----------
    image : ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this low to detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this high to detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect less prominent blobs.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.

    Returns
    -------
    A : (n, 3) ndarray
        A 3d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel of the Hessian Matrix whose
        determinant detected the blob.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian

    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

     
    Notes
    -----
    The radius of each blob is approximately `sigma`.
    Computation of Determinant of Hessians is independent of the standard
    deviation. Therefore detecting larger blobs won't take more time. In
    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation
    of Gaussians for larger `sigma` takes more time. The downside is that
    this method can't be used for detecting blobs of radius less than `3px`
    due to the box filters used in the approximation of Hessian Determinant.
    """
