import numpy as np
from scipy.integrate import odeint

def active_model(y, t, k):
    """
    Defines an adapted SEIR model for two-stage active immunisation (vaccination)
    -------
    Inputs
    -------
    y: list
        List of compartments
    t: list
        Time interval
    k: list
        List of model parameter values
    -------
    Outputs
    -------
    dzdt: list
        List of governing equations
    """   

    # Unpack variables and parameter values
    S, S_v0, S_v1, S_v2, E, I_m, I_s, R = y
    q, q_v1, q_v2, beta_m, beta_v1m, beta_v2m, beta_s, beta_v1s, beta_v2s, gamma_m, gamma_s, epsilon_1, epsilon_2, c, a, p, N  = k

    # Define governing equations for each compartment
    dSdt = -(q*c + (beta_s*I_s + beta_m*I_m)/N)*S
    dS_v0dt = q*c*S - (q_v1 + (beta_s*I_s + beta_m*I_m)/N)*S_v0
    dS_v1dt = q_v1*S_v0 - (q_v2 + (1-epsilon_1)*(beta_v1s*I_s + beta_v1m*I_m)/N)*S_v1
    dS_v2dt = q_v2*S_v1 - ((1-epsilon_2)*(beta_v2s*I_s + beta_v2m*I_m)/N)*S_v2
    dEdt = (beta_s*I_s + beta_m*I_m)*(S + S_v0)/N + ((1-epsilon_1)*(beta_v1s*I_s + beta_v1m*I_m)/N)*S_v1 + ((1-epsilon_2)*(beta_v2s*I_s + beta_v2m*I_m)/N)*S_v2 - a*E
    dI_mdt = a*p*E - gamma_m*I_m
    dI_sdt = a*(1-p)*E - gamma_s*I_s
    dRdt = gamma_m*I_m + gamma_s*I_s

    # Define output
    dydt = [dSdt, dS_v0dt, dS_v1dt, dS_v2dt, dEdt, dI_mdt, dI_sdt, dRdt]
    return dydt

def passive_model(y, t, k):
    """
    Defines an adapted SEIR model for passive immunisation (prophylaxis)
    -------
    Inputs
    -------
    y: list
        List of compartments
    t: list
        Time interval
    k: list
        List of model parameter values
    -------
    Outputs
    -------
    dzdt: list
        List of governing equations
    """      
 
    # Unpack variables and parameters
    S, S_s, E, I_m, I_s, R = y
    q, beta_m, beta_m2, beta_s, beta_s2, gamma_m, gamma_s, epsilon_ss, c, a, p, N  = k

    # Define governing equations for each compartment
    dSdt = -(q*c + (beta_s*I_s + beta_m*I_m)/N)*S
    dS_sdt = q*c*S - ((1-epsilon_ss)*(beta_s2*I_s + beta_m2*I_m)/N)*S_s
    dEdt = (beta_s*I_s + beta_m*I_m)*S/N + ((1-epsilon_ss)*(beta_s2*I_s + beta_m2*I_m)/N)*S_s -a*E
    dI_mdt = a*p*E - gamma_m*I_m
    dI_sdt = a*(1-p)*E - gamma_s*I_s
    dRdt = gamma_m*I_m + gamma_s*I_s

    # Define output
    dydt = [dSdt, dS_sdt, dEdt, dI_mdt, dI_sdt, dRdt]
    return dydt