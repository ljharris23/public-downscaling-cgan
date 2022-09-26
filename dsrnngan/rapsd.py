# BSD 3-Clause License

# Copyright (c) 2019, PySteps developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import matplotlib.pyplot as plt


def compute_centred_coord_array(M, N):
    """Compute a 2D coordinate array, where the origin is at the center.
    Parameters
    ----------
    M : int
      The height of the array.
    N : int
      The width of the array.
    Returns
    -------
    out : ndarray
      The coordinate array.
    Examples
    --------
    >>> compute_centred_coord_array(2, 2)
    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))
    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2): int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2): int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2): int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2): int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def rapsd(
    field, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs
):
    """Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.
    Parameters
    ----------
    field: array_like
        A 2d array of shape (m, n) containing the input field.
    fft_method: object
        A module or object implementing the same methods as numpy.fft and
        scipy.fftpack. If set to None, field is assumed to represent the
        shifted discrete Fourier transform of the input field, where the
        origin is at the center of the array
        (see numpy.fft.fftshift or scipy.fftpack.fftshift).
    return_freq: bool
        Whether to also return the Fourier frequencies.
    d: scalar
        Sample spacing (inverse of the sampling rate). Defaults to 1.
        Applicable if return_freq is 'True'.
    normalize: bool
        If True, normalize the power spectrum so that it sums to one.
    Returns
    -------
    out: ndarray
      One-dimensional array containing the RAPSD. The length of the array is
      int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: ndarray
      One-dimensional array containing the Fourier frequencies.
    References
    ----------
    :cite:`RC2011`
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])  # noqa

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


def plot_spectrum1d(
    fft_freq,
    fft_power,
    x_units=None,
    y_units=None,
    wavelength_ticks=None,
    color="k",
    lw=1.0,
    label=None,
    ax=None,
    **kwargs,
):
    """
    Function to plot in log-log a radially averaged Fourier spectrum.
    Parameters
    ----------
    fft_freq: array-like
        1d array containing the Fourier frequencies computed with the function
        :py:func:`pysteps.utils.spectral.rapsd`.
    fft_power: array-like
        1d array containing the radially averaged Fourier power spectrum
        computed with the function :py:func:`pysteps.utils.spectral.rapsd`.
    x_units: str, optional
        Units of the X variable (distance, e.g. "km").
    y_units: str, optional
        Units of the Y variable (amplitude, e.g. "dBR").
    wavelength_ticks: array-like, optional
        List of wavelengths where to show xticklabels.
    color: str, optional
        Line color.
    lw: float, optional
        Line width.
    label: str, optional
        Label (for legend).
    ax: Axes, optional
        Plot axes.
    Returns
    -------
    ax: Axes
        Plot axes
    """
    # Check input dimensions
    n_freq = len(fft_freq)
    n_pow = len(fft_power)
    if n_freq != n_pow:
        raise ValueError(
            f"Dimensions of the 1d input arrays must be equal. {n_freq} vs {n_pow}"
        )

    if ax is None:
        ax = plt.subplot(111)

    # Plot spectrum in log-log scale
    ax.plot(
        10 * np.log10(fft_freq[np.where(fft_freq > 0.0)]),
        10 * np.log10(fft_power[np.where(fft_freq > 0.0)]),
        color=color,
        linewidth=lw,
        label=label,
        **kwargs,
    )

    # X-axis
    if wavelength_ticks is not None:
        wavelength_ticks = np.array(wavelength_ticks)
        freq_ticks = 1 / wavelength_ticks
        ax.set_xticks(10 * np.log10(freq_ticks))
        ax.set_xticklabels(wavelength_ticks)
        if x_units is not None:
            ax.set_xlabel(f"Wavelength [{x_units}]")
    else:
        if x_units is not None:
            ax.set_xlabel(f"Frequency [1/{x_units}]")

    # Y-axis
    if y_units is not None:
        # { -> {{ with f-strings
        power_units = fr"$10log_{{ 10 }}(\frac{{ {y_units}^2 }}{{ {x_units} }})$"
        ax.set_ylabel(f"Power {power_units}")

    return ax
