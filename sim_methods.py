"""
Created on Fri 25 Jan 2019

@author: R Carthigesan

Module implementing methods required for simulation of dynamics
"""

# Import required libraries
import numpy as np
import math

# Hitchhiking


def hitch_kern(dh=10 ** (-6)):
    """Generates jump kernel cumulative distribution function

        Args:
            dh (float, optional): Discretisation scale in jump kernel distribution

        Returns:
            h (float): Hitchhiking ump size"""

    e = 0.001  # 1/exp(5.0)
    h_list = np.linspace(e, 1.0, 1.0 / dh)
    cumulative_h = {}

    for h in h_list:
        cumul = round(1.0 - (math.log(h) / math.log(e)), 3)
        cumulative_h[cumul] = h

    return cumulative_h


def hitchhike(freq, cumulative_h):
    """Stochastic function returning the jump in frequency due to a hitchhiking event.

        Args:
            freq (float): Frequency value
            cumulative_h (dict): Jump kernel cumulative distribution function

        Returns:
            hitchhiking_step (float): Change in frequency due to hitchhiking"""

    eta = np.random.random()
    h = cumulative_h[round(eta, 3)]

    if np.random.uniform(0, 1) < freq:  # if beneficial mutation lands on genome with genotype in question
        hitchhiking_step = h * (1 - freq)

    else:  # if beneficial mutation lands on genome with alternative genotype to that in question
        hitchhiking_step = -1 * freq * h

    return hitchhiking_step


# Mutation


def mutate(freq, mu, dt):
    """Returns the change in frequency due to mutation.

        Args:
            freq (float): Frequency value
            mu (float): Mutation rate
            dt (float): Time step length

        Returns:
            mutation_step (float): Change in frequency due to mutation"""

    mutation_step = mu * (1.0 - 2.0 * freq) * dt

    return mutation_step


# Drift


def drift(freq, t, t_bottleneck, n_init, growth_time):
    """Returns the change in frequency due to drift, following the Wright-Fisher model.

        Args:
            freq (float): Frequency value
            t (int): Time
            t_bottleneck (int): Time at which population growth starts
            n_init (int): Initial population size
            growth_time (int): Time for population to grow by a factor of e

        Returns:
            drift_jump (float): Change in frequency due to drift
            n (int): The current population size"""

    if t < t_bottleneck:
        n = n_init
    else:
        n = n_init * np.exp((t - t_bottleneck) / growth_time)

    drift_jump = ((1 / n) * np.random.binomial(n, freq)) - freq

    return drift_jump, int(n)


# Simulate dynamics


def dynamics(times, dt, n_init, mu, gamma, hitchhiking_kernel, sampling_period=1, t_bottleneck=int(1e20),
             growth_time=1, freq_init=0.0):
    """Calculates dynamics of neutral site frequency subject to drift, mutation and hitchhiking.

            Args:
                times (numpy ndarray): Times to simulate
                dt (int): Time step length
                n_init (int): Initial population size
                mu (float): Rate of mutations
                gamma (float): Rate at which beneficial; mutations come in that are linked to the neutral being observed
                hitchhiking_kernel (dict): The hitchhiking kernel
                sampling_period (int, optional): Defines coarseness of sampled results. For example, if it is set to 5
                                                 then we sample the frequency for every fifth time step (default = 1)
                t_bottleneck (int, optional): Time at which population growth starts
                                              (default is very large, i.e. no growth)
                growth_time (int): Time for population to grow by a factor of e (default = 1)
                freq_init (float, optional): Initial frequency (default = 0)

            Returns:
                frequencies (list): List of frequency values at each sampled time"""

    frequencies = np.zeros(times.size, dtype=float)
    frequencies[0] = freq_init

    # gamma = 10**3/n_init

    # t_hitch = int(np.random.exponential(1.0 / gamma))  # generate time for first hitchhiking event

    for i, t in enumerate(times[1:]):

        drift_jump, n = drift(frequencies[i], t, t_bottleneck, n_init, growth_time)

        frequencies[i+1] = frequencies[i] + mutate(frequencies[i], mu, dt) + drift_jump  # mutation and drift

        # if t > t_hitch:  # check if a hitchhiking event is due
        #     frequencies[i + 1] += hitchhike(frequencies[i], hitchhiking_kernel)
        #     gamma = 10**3/n
        #     t_hitch = t + int(np.random.uniform(0, (1 / gamma)))  # generate time for next hitchhiking event

        frequencies[i + 1] = np.clip(frequencies[i+1], 0.0, 1.0)  # handle unphysical frequencies

    frequencies = frequencies[::sampling_period]  # comment this line out if using sampling_period = 1 for efficiency

    return frequencies


# Miscellaneous


def get_random_colour(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [np.random.uniform() for i in [1, 2, 3]]]


def colour_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


def generate_new_colour(existing_colours, pastel_factor=0.5):
    max_distance = None
    best_colour = None
    for i in range(0, 100):
        colour = get_random_colour(pastel_factor=pastel_factor)
        if not existing_colours:
            return colour
        best_distance = min([colour_distance(colour, c) for c in existing_colours])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_colour = colour
    return best_colour


def randomvariate(pdf, n, mu, t, n_samples=1000, xmin=0, xmax=1):
    """
   Rejection method for random number generation
   ===============================================
   Uses the rejection method for generating random numbers derived from an arbitrary
   probability distribution. For reference, see Bevington's book, page 84. Based on
   rejection*.py.

   Usage:
   >>> randomvariate(pdf, n_samples, xmin, xmax)
    where
    pdf : probability distribution function from which you want to generate random numbers
    n: population size
    mu: mutation rate ber pase pair
    t: time
    n_samples : desired number of random values
    xmin,xmax : range of random numbers desired

   Returns:
    the sequence (ran,ntrials) where
     ran : array of shape N with the random variates that follow the input P
     ntrials : number of trials the code needed to achieve N

   Here is the algorithm:
   - generate x' in the desired range
   - generate y' between Pmin and Pmax (Pmax is the maximal value of your pdf)
   - if y'<P(x') accept x', otherwise reject
   - repeat until desired number is achieved

   Rodrigo Nemmen
   Nov. 2011
    """

    # Calculates the minimal and maximum values of the PDF in the desired
    # interval. The rejection method needs these values in order to work
    # properly.
    x = np.linspace(xmin, xmax, 1000)
    y = pdf(x, n, mu, t)
    pmin = 0.
    pmax = y.max()

    # Counters
    naccept = 0
    ntrial = 0

    # Keeps generating numbers until we achieve the desired n
    ran = []  # output list of random numbers
    while naccept < n_samples:
        x = np.random.uniform(xmin, xmax)  # x'
    y = np.random.uniform(pmin, pmax)  # y'

    if y < pdf(x):
        ran.append(x)
        naccept += 1
    ntrial += 1

    ran = np.asarray(ran)

    return ran, ntrial
