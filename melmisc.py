import numpy as np
from scipy.signal import lfilter, lfiltic, butter

def hz2mel(F):
    """
    Transforms Hz to Mel scale using formulat from https://en.wikipedia.org/wiki/Mel_scale
    
    Parameters
    ----------
    F : float
        frequency in Hz

    Returns
    -------
    m : float
        frequency on Mel scale
    """
    return 2595. * np.log10( 1. + F / 700. )

def mel2hz(m):
    """
    Transforms frequency on Mel scale to Hz using formulat from https://en.wikipedia.org/wiki/Mel_scale
    
    Parameters
    ----------
    m : float
        frequency on Mel scale

    Returns
    -------
    F : float
        frequency in Hz
    """
    return 700. * ( 10.** (m/2595.) - 1. )

def meluniform(F_min = 0, F_max = 44100, num = 50):
    """
    Generates an array of frequencies uniformly distributed on Mel scale
    
    Parameters
    ----------
    F_min : float, optional
        maximal frequency in Hz
    F_max : float, optional
        maximal frequency in Hz
    num : int, optional
        number of elements

    Returns
    -------
    F : numpy array
        frequencies in Hz
    """
    return mel2hz( np.linspace(start = hz2mel(F_min), stop = hz2mel(F_max), num = num) )


def spec_tri_window(n, center, left = None, right = None):
    """
    Assymetrical triangular window
    
    Parameters
    ----------
    samples : int
        total number of samples in the domain
    center : int 
        center of the window
    left : int, optional
        left point of the window; can be ommited if center = 0
    right: int, optional
        right point of the window; can be ommited if center = samples

    Returns
    -------
    window : numpy array
        window is zero everywhere except for the domain between left and right
    """
    window = np.zeros((n,), dtype = float)
    if left is not None:
        window[left:center+1] = np.linspace(0, 1, center-left+1)
    elif center != 0:
        raise ValueError('left can be ommited only when center = 0')
    if right is not None:
        try:
            window[center:right+1] = np.linspace(1, 0, right-center+1)
        except ValueError:
            window[center:right+1] = np.linspace(1, 0, right-center+1)[ :window[center:right+1].shape[0] ]
    elif center != n-1:
        raise ValueError('right can be ommited only when center = n-1')
    return window

def mel_scale_windows(n, F_max, num, F_max_mel = None):
    """
    List of triangular windows to convert spectrum from Hz to Mel scale
    
    Parameters
    ----------
    n : int
        number of samples in the spectrum
    F_max : float
        maximal frequency in the spectrum
    num : int
        number of points on Mel scale

    Returns
    -------
    windows : list of numpy arrays
        each element defines a window for a certain central frequency on Mel scale
    """
    if F_max_mel is None:
        F_max_mel = F_max
    mel = meluniform(F_min = 0, F_max = F_max_mel, num = num + 1)
    windows = []
    centers = np.round( mel / (float(F_max) / n) )
    left = None
    center = centers[0]
    for right in centers[1:]:
        window = spec_tri_window(n = n, center = center, left = left, right = right)
        left, center = center, right
        windows.append(window)
    return windows

def mel_scale_butter(F, F_max, num):
    mel = meluniform(F_min = 0, F_max = F_max, num = 2 * num + 1)
    filters = [];
    b, a = butter(2, mel[1] / F);
    filters.append( (b,a) )
    for cnt in range(3, len(mel)-2, 2):
        b, a = butter(2, [mel[cnt-1], mel[cnt+1]] / F, btype = 'band');
        filters.append( (b,a) )
    b, a = butter(2, mel[-2] / F);
    filters.append( (b,a) )
        
