import numpy as np
import csv

"""
Empirical functions for calculating models of sediment dynamics
"""


def get_ew(U, Ch, R, g, ewc, umin=0.01, out=None):
    """ calculate entrainment coefficient of ambient water to a turbidity
        current layer

        Parameters
        ----------
        U : ndarray, float
           Flow velocities of a turbidity current.
        Ch : ndarray, float
           Flow height times sediment concentration of a turbidity current.
        R : float
           Submerged specific density of sediment
        g : float
           gravity acceleration
        umin: float
           minimum threshold value of velocity to calculate water entrainment

        out : ndarray
           Outputs

        Returns
        ---------
        e_w : ndarray, float
           Entrainment coefficient of ambient water

    """
    if out is None:
        out = np.zeros(U.shape)

    Ri = np.zeros(U.shape)
    flowing = np.where(U > umin)
    Ri[flowing] = R * g * Ch[flowing] / U[flowing] ** 2
    out = (0.075 / np.sqrt(1 + 718.0 * Ri**2.4)) * ewc # Def
    # out = (0.075 / (1 + 718.0 * Ri**3.4)**0.33) * ewc # Traer et al. (2012)
    # out = 0.075 / np.sqrt(1 + 718.0 * Ri ** 2.4) # Parker et al. (1987)
    # out = 0.075 / (1 + 718.0 * Ri ** 0.72)**1.5 # Traer et al. (2015) ew3
    # out = 0.00153 / (1e-10 + 0.0204 * Ri) # Fukushima et al. (1985)

    return out

def get_det_rate(ws, Ch_i, h, det_factor=1.0, out=None):
    """Calculate rate of detrainment caused by sediment
      settling

      The detrainment rate at the flow interface is assumed
      to be proportional to the sediment settling rate.
      The default value for the proportionality factor is 1.0,
      but Salinas et al. (2019) estimate that 3.05 is
      an appropriate value based on DNS calculations.

    Parameters
    ----------
    ws : ndarray, float
       sediment settling rate
    C_i: ndarray(2d), float
        Sediment concentration for the ith grain size class
    det_factor: float
        Factor for detrainment rate. The default value is
        1.0.

    out : ndarray
       Outputs

    Returns
    ---------
    e_d : ndarray, float
       Detrainment coefficient of water (positive when
       fluid exits the flow)
    """

    if out is None:
        out = np.zeros(h.shape)

    eps = 1.0e-15
    C_i = Ch_i / h + eps

    out[:] = det_factor * np.sum((ws * C_i / np.sum(C_i, axis=0)), axis=0)

    return out


def get_ws(R, g, Ds, nu):
    """Calculate settling velocity of sediment particles
        on the basis of Ferguson and Church (1982)

    Return
    ------------------
    ws : settling velocity of sediment particles [m/s]

    """

    # Coefficients for natural sands
    C_1 = 18.0
    C_2 = 1.0

    ws = R * g * Ds**2 / (C_1 * nu + (0.75 * C_2 * R * g * Ds**3) ** 0.5)

    return ws


def get_es(R, g, Ds, nu, u_star, Fr, bed_active_layer, p_coef, camax, function="GP1991field", out=None):
    """ Calculate entrainment rate of basal sediment to suspension using
        empirical functions proposed by Garcia and Parker (1991),
        van Rijn (1984), or Dorrell (2018)

        Parameters
        --------------
        R : float
            submerged specific density of sediment (~1.65 for quartz particle)
        g : float
            gravity acceleration
        Ds : float
            grain size
        nu : float
            kinematic viscosity of water
        u_star : ndarray
            flow shear velocity
        Fr : ndarray
            froude number
        function : string, optional
            Name of emprical function to be used.

            'GP1991exp' is a function of Garcia and Parker (1991)
             in original form. This is suitable for experimental scale.

            'GP1991field' is Garcia and Parker (1991)'s function with
            a coefficient (0.1) to limit the entrainment rate. This is suitable
            for the natural scale.

        out : ndarray
            Outputs (entrainment rate of basal sediment)

        Returns
        ---------------
        out : ndarray
            dimensionless entrainment rate of basal sediment into
            suspension

    """
    if out is None:
        out = np.zeros([len(Ds), len(u_star)])

    if function == "GP1991field":
        _gp1991(R, g, Ds, nu, u_star, p=p_coef, alpha = 0.6, beta=5.0, out=out)
    if function == "GP1991exp":
        _gp1991(R, g, Ds, nu, u_star, p=1.0, alpha = 0.6, beta=5.0, out=out)
    if function == "Traer2012":
        _gp1991(R, g, Ds, nu, u_star, p=p_coef, alpha = 0.68, beta=2.8, out=out)
    if function== 'wright_and_parker(2004)':
        _wright_and_parker(R, g, Ds, nu, u_star, sigma=0.52, slope_inside=2.4*10**-5, out=None) #slope_inside=2.4*10**-5
    if function == "Leeuw2020P3":
        _l2020p3(R, g, Ds, nu, u_star, Fr, out=out)
    if function == "Leeuw2020P2":
        _l2020p2(R, g, Ds, nu, u_star, Fr, out=out)
    if function == "Leeuw2020P1":
        _l2020p1(R, g, Ds, nu, u_star, out=out)
    if function == "NRv1":
        _NRv1(R, g, Ds, nu, u_star, p=p_coef, beta=3.0, out=out)
    if function == "NRv2":
        _NRv2(R, g, Ds, nu, u_star, bed_active_layer, p=p_coef, beta=2.8, camax=camax, out=out)
    if function == "NRv3":
        _NRv3(R, g, Ds, nu, u_star, p=p_coef, beta=3.0, camax=camax, out=out)

    return out

def _NRv1(R, g, Ds, nu, u_star, p, beta=3.0, out=None):

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    # basic parameters
    ws = get_ws(R, g, Ds, nu)

    # calculate subordinate parameters
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    sus_index = u_star / ws
    D50 = np.sum(bed_active_layer*Ds, axis=0) #1.2e-4
    D50_phi = -np.log2(D50*1000)
    Ds_phi = -np.log2(Ds*1000)
    sigma = np.sqrt(np.sum(bed_active_layer * (Ds_phi - D50_phi) ** 2, axis=0)) #1.0
    kshi = 1-0.288*sigma
    a = 1.3 * 10**-7
    print("Ds",Ds)
    print("D50",D50)
    print("sigma",sigma)
    # coefficients for calculation
    alpha_1=[]
    alpha_2=[]
    for i in Rp:
        if i >= 2.36:
            alpha_1=np.append(alpha_1, [1.0])
            alpha_2=np.append(alpha_2, [0.6])
        elif 1 <= i < 2.36:
            alpha_1=np.append(alpha_1, [0.586])
            alpha_2=np.append(alpha_2, [1.23])
        else:
            alpha_1=np.append(alpha_1, [0.43])
            alpha_2=np.append(alpha_2, [1.16])
    alpha_1 = alpha_1.reshape(len(Ds),1)
    alpha_2 = alpha_2.reshape(len(Ds),1)

    # alpha_1=1
    # alpha_2=0.6

    # calculate entrainment rate
    Z = kshi * alpha_1 * sus_index * Rp ** alpha_2
    out[:, :] = p * a * Z ** beta / (1 + (a / 0.3) * Z ** beta) * (Ds/D50)**0.2

    return out

def _NRv2(R, g, Ds, nu, u_star, bed_active_layer, p, beta=2.8, camax=0.3, out=None):

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    # basic parameters
    ws = get_ws(R, g, Ds, nu)

    # calculate subordinate parameters
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    sus_index = u_star / ws
    D50 = np.sum(bed_active_layer*Ds, axis=0) #1.2e-4
    D50_phi = -np.log2(D50*1000)
    Ds_phi = -np.log2(Ds*1000)
    sigma = np.sqrt(np.sum(bed_active_layer * (Ds_phi - D50_phi) ** 2, axis=0)) #1.0
    kshi = 1 -0.288 * sigma
    a = 1.3 * 10**-7 #7*10**-6 #

    # coefficients for calculation
    alpha_1=1
    alpha_2=0.678 #0.6 #

    # calculate entrainment rate
    Z = kshi * alpha_1 * sus_index * Rp ** alpha_2
    # Z = alpha_1 * sus_index * Rp ** alpha_2
    out[:, :] = p * a * Z ** beta / (1 + (a / camax) * Z ** beta) * (Ds/D50)**0.2

    return out

# def _NRv3(R, g, Ds, nu, u_star, p, beta=2.8, camax=0.3, out=None):
#
#     if out is None:
#         out = np.zeros([len(Ds), u_star.shape])
#
#     # basic parameters
#     ws = get_ws(R, g, Ds, nu)
#
#     # calculate subordinate parameters
#     Rp = np.sqrt(R * g * Ds) * Ds / nu
#     sus_index = u_star / ws
#     a = 1.3 * 10**-7
#
#     # coefficients for calculation
#     alpha_1=1
#     alpha_2=0.68
#
#     # calculate entrainment rate
#     Z = alpha_1 * sus_index * Rp ** alpha_2
#     out[:, :] = p * a * Z ** beta / (1 + (a / camax) * Z ** beta)
#
#     return out

def _gp1991(R, g, Ds, nu, u_star, p, alpha = 0.6, beta=5.0, out=None):
    """ Calculate entrainment rate of basal sediment to suspension
        Based on Garcia and Parker (1991)

        Parameters
        --------------
        u_star : ndarray
            flow shear velocity
        out : ndarray
            Outputs (entrainment rate of basal sediment)

        Returns
        ---------------
        out : ndarray
            dimensionless entrainment rate of basal sediment into
            suspension
    """

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    # basic parameters
    ws = get_ws(R, g, Ds, nu)

    # calculate subordinate parameters
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    sus_index = u_star / ws

    # coefficients for calculation
    a = 1.3 * 10**-7

    # calculate entrainment rate
    Z = sus_index * Rp**alpha
    out[:, :] = p * a * Z**beta / (1 + (a / 0.3) * Z**beta)

    # with np.errstate(invalid='ignore', divide='ignore'):
    #     out_res = np.block([np.max(out, axis=1), np.sum(out, axis=1)/np.count_nonzero(out, axis=1),
    #                         np.max(u_star), np.sum(u_star)/np.count_nonzero(u_star)])
    # with open('/mnt/d/turb2d/es_trear2012.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_res)

    return out

def _wright_and_parker(R, g, Ds, nu, u_star, sigma, slope_inside, out=None):

    if out is None:
        out = np.zeros((len(Ds), len(u_star)))

    ws = get_ws(R, g, Ds, nu)
    a = 7.8 * 10**-7
    me = 0.07
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    kshi = 1-0.288*sigma
    Ds50 = np.median(Ds)
    sus_index = u_star / ws

    alpha_1=[]
    alpha_2=[]
    for i in Rp:
        if i > 2.36:
            alpha_1=np.append(alpha_1, [1.0])
            alpha_2=np.append(alpha_2, [0.6])
        elif i <= 2.36:
            alpha_1=np.append(alpha_1, [0.586])
            alpha_2=np.append(alpha_2, [1.23])
    alpha_1 = alpha_1.reshape(len(Ds),1)
    alpha_2 = alpha_2.reshape(len(Ds),1)

    # ks = 6.0 * Ds50
    # Hsk = ((8.1**6 * g**3 * slope_inside**3)/((u_star**2)**6 * ks))**1/4
    # u_star_sk = np.sqrt(g*Hsk*slope_inside)

    Z = alpha_1 * kshi * sus_index * Rp**alpha_2 * slope_inside**0.08 * (Ds/Ds50)**0.2
    out[:, :] = me * a * Z**5 / (1 + (a / 0.3) * Z**5)
    with np.errstate(invalid='ignore', divide='ignore'):
        out_res = np.block([np.max(out, axis=1), np.sum(out, axis=1)/np.count_nonzero(out, axis=1),
                            np.max(u_star), np.sum(u_star)/np.count_nonzero(u_star)])
    with open('/mnt/d/turb2d/es_wright_and_parker.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(out_res)

    return out

# def _l2020p3(R, g, Ds, nu, u_star, Fr, out=None):
#
#     if out is None:
#         out = np.zeros((len(Ds), len(u_star)))
#
#     ws = get_ws(R, g, Ds, nu)
#     a = 1.30 * 10**-3
#     e1, e2, e3 = 1.398, 1.576, -0.521
#     Rep = np.sqrt(R * g * Ds) * Ds / nu
#     sus_index = u_star / ws
#
#     with np.errstate(invalid='ignore', divide='ignore'):
#         out[:, :] = a * sus_index**e1 * Fr**e2 * Rep**e3
#         out[np.isnan(out)] = 0
#         out_res = np.block([np.max(out, axis=1), np.sum(out, axis=1)/np.count_nonzero(out, axis=1),
#                             np.max(u_star), np.sum(u_star)/np.count_nonzero(u_star), np.max(Fr), np.sum(Fr)/np.count_nonzero(Fr)])
#
#     with open('/mnt/d/turb2d/es_l2020p3.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(out_res)
#
#     return out
#
# def _l2020p2(R, g, Ds, nu, u_star, Fr, out=None):
#
#     if out is None:
#         out = np.zeros((len(Ds), len(u_star)))
#
#     ws = get_ws(R, g, Ds, nu)
#     a = 7.42 * 10**-4
#     e1, e2 = 1.687, 1.823
#     sus_index = u_star / ws
#
#     with np.errstate(invalid='ignore', divide='ignore'):
#         out[:, :] = a * sus_index**e1 * Fr**e2
#         out[np.isnan(out)] = 0
#         out_res = np.block([np.max(out, axis=1), np.sum(out, axis=1)/np.count_nonzero(out, axis=1),
#                             np.max(u_star), np.sum(u_star)/np.count_nonzero(u_star), np.max(Fr), np.sum(Fr)/np.count_nonzero(Fr)])
#
#     with open('/mnt/d/turb2d/es_l2020p2.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(out_res)
#
#     return out
#
# def _l2020p1(R, g, Ds, nu, u_star, out=None):
#
#     if out is None:
#         out = np.zeros((len(Ds), len(u_star)))
#
#     ws = get_ws(R, g, Ds, nu)
#     a = 3.70 * 10**-5
#     e1 = 1.43
#     sus_index = u_star / ws
#
#     with np.errstate(invalid='ignore', divide='ignore'):
#         out[:, :] = a * sus_index**e1
#         out[np.isnan(out)] = 0
#         out_res = np.block([np.max(out, axis=1), np.sum(out, axis=1)/np.count_nonzero(out, axis=1),
#                             np.max(u_star), np.sum(u_star)/np.count_nonzero(u_star)])
#
#     with open('/mnt/d/turb2d/es_l2020p1.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(out_res)
#
#     return out

def get_bedload(u_star, Ds, R=1.65, g=9.81, function="MPM", out=None):
    """Get bedload discharge from empirical formulation

       Parameters
       ------------------------------
       u_star: 1d ndarray
          friction velocity

       Ds: 1d ndarray
          grain diameters

       R: float, optional
          Submerged specific density of sediment particles.
          Default is 1.65

       g: float, optional
          gravity acceleration.
          Default is 9.81

       function: str, optional
          Function name for prediting bedload discharge
          Default is "MPM". Other options are:
          "WP2006": Wong and Parker (2006)

       out: 1d ndarray
          Outputs (1d array of sediment bedload discharge)

       Returns
       ---------------
       out : ndarray
         1d array of sediment bedload discharge

    """

    if out is None:
        out = np.zeros([len(Ds), len(u_star)])

    if function == "MPM":
        _MPM(u_star, Ds, R, g, a=8.0, b=1.5, out=out)
    elif function == "WP2006":
        _MPM(u_star, Ds, R, g, a=4.93, b=1.6, out=out)
    else:
        _MPM(u_star, Ds, R, g, a=8.0, b=1.5, out=out)

    return out

def _MPM(u_star, Ds, R=1.65, g=9.81, a=8.0, b=1.5, out=None):
    """Bedload prediction by Meyer=Peter and
       Muller (1948)-type equations

       Parameters
       ------------------------------
       u_star: 1d ndarray
          friction velocity

       Ds: 1d ndarray
          grain diameters

       R: float, optional
          Submerged specific density of sediment particles.
          Default is 1.65

       g: float, optional
          gravity acceleration.
          Default is 9.81

       a: float, optional
          coefficient used in the MPM equation
          Default is 8.0

       b: float, optional
          exponent used in the MPM-type equation

       out: 1d ndarray
          Outputs (1d array of sediment bedload discharge)

       Returns
       ---------------
       out : ndarray
         1d array of sediment bedload discharge

    """

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    tau_c = 0.047

    tau_star_c = u_star * u_star / (R * g * Ds) - tau_c

    tau_star_c = np.where(
        tau_star_c > 0.0,
        tau_star_c,
        0.0
    )

    out[:, :] = a * tau_star_c ** b * np.sqrt(R * g * Ds ** 3)

    return out
