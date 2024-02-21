import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
from matplotlib import patches
import matplotlib.colors as mcolors
import fractions

#%%

plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

#%%

'''Define functions to calculate parameters'''

def Phi(r):
    return -GM / (b + np.sqrt(b**2 + r**2))

def v_c(r):
    a = np.sqrt(b**2 + r**2)
    return (GM * r**2 / ((b + a)**2 * a))**0.5

def H(J_r, L):
    return -0.5*GM**2 / (J_r + 0.5*(L + np.sqrt(L**2 + 4*GM*b)))**2

def calc_J_r(E, L):
    return GM/np.sqrt(-2*E) - 0.5*(L + np.sqrt(L**2 + 4*GM*b))

def calc_Omega_r(J_r, L):
    return GM**2 / (J_r + 0.5*(L + np.sqrt(L**2 + 4*GM*b)))**3

def calc_Omega_phi(J_r, L, J_phi):
    return 0.5*(1 + L/np.sqrt(L**2 + 4*GM*b)) * calc_Omega_r(J_r, L) * np.sign(J_phi)

def H_res(L, J_phi, l, m):
    A = -0.5*(GM*Omega_b)**(2/3)
    return A * (0.5*(np.sign(J_phi) + L*np.sign(J_phi)/np.sqrt(L**2 + 4*GM*b)) + l/m)**(-2/3)

def vr_res(r, L, J_phi, l, m, theta_r=0):
    
    sign = np.sign(np.pi - theta_r % (2*np.pi))
    
    return sign*(2*(H_res(L, J_phi, l, m) - Phi(r)) - L**2/r**2)**0.5


# Choose Solar radius and local circular velocity
r0 = 8.2
v0 = 238

# Choose scale radius and bar pattern speed
b = 3
Omega_b = 35
bar_angle = -30 * np.pi/180

# Calculate mass required to satisfy above requirements
a = np.sqrt(b**2 + r0**2)
GM = v0**2 * ((b+a)**2 * a)/r0**2
M = (GM*(u.km/u.s)**2*u.kpc / G).to(u.Msun)

#%%

'''Plot E/L_z space and action space for resonant orbits'''

Lz_grid = np.linspace(-3500, 3500, 8000)
E_disc = H(0, Lz_grid)#Phi(r) + 0.5 * v_c(r)**2


# Calculate Lz and E at Sun's radius
Lz0 = r0 * v_c(r0)
E0 = H(0, Lz0)

l_grid = [0, 1, 2, 3, 4]
m_grid = [2, 2, 2, 2, 2]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']
ls = ['-', '--','-.', ':', '-']

fig, axs = plt.subplots(2, figsize=(4,8), sharex=True)
plt.subplots_adjust(hspace=0)

for i in range(len(l_grid)):
    E_res = H_res(abs(Lz_grid), Lz_grid, l_grid[i], m_grid[i])
    J_r_res = calc_J_r(E_res, abs(Lz_grid))
    
    label = '$l/m=$ '+str(fractions.Fraction(l_grid[i], m_grid[i]))
    
    axs[0].plot(Lz_grid, E_res/1e5, color=colors[i], label=label, ls=ls[i], lw=2)
    axs[1].plot(Lz_grid, J_r_res, color=colors[i], ls=ls[i], lw=2)


axs[0].plot(Lz_grid, E_disc/1e5, ls='--', c='gray', zorder=3)
axs[0].plot(-Lz_grid, E_disc/1e5, ls='--', c='gray', zorder=3)


axs[0].fill_between(Lz_grid, -1, E_disc/1e5, fc='w', zorder=2)
axs[0].fill_between(-Lz_grid, -1, E_disc/1e5, fc='w', zorder=2)


axs[0].scatter(Lz0, E0/1e5, marker='o', c='w', edgecolor='k', s=100, zorder=4)
axs[0].scatter(Lz0, E0/1e5, marker='o', c='k', s=5, zorder=4)

axs[0].legend(loc=2, fontsize=12, ncol=2).set_zorder(3)

axs[0].set_xlim(min(Lz_grid), max(Lz_grid))
axs[0].set_ylim(-1, 0.1)
axs[1].set_ylim(0)

# plt.axhline(E_sol)

axs[1].set_xlabel('$J_\phi$ [kpc km/s]', fontsize=16)
axs[0].set_ylabel('$E$ [$10^5$ (km/s)$^2$]', fontsize=16)
axs[1].set_ylabel('$J_r$ [kpc km/s]', fontsize=16)

for ax in axs.flat:
    ax.tick_params(top=True, bottom=True, right=True, left=True, direction='in', labelsize=12)

plt.show()
plt.close()

#%%

'''Plot condition for existence of resonant orbits'''

Omega_b = 35
I = 0 * np.pi/180

L = abs(Lz_grid) / np.cos(I)

l_m_min_Omega_r = -0.5*(1 + L/np.sqrt(L**2 + 4*GM*b))*np.sign(Lz_grid)

l_m_min_J_r = Omega_b/GM**2 * (0.5*(L + np.sqrt(L**2+4*GM*b)))**3 - 0.5*(1 + L/np.sqrt(L**2 + 4*GM*b))*np.sign(Lz_grid)

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(Lz_grid, l_m_min_Omega_r, c='k')
ax.plot(Lz_grid, l_m_min_J_r, c='k', ls='--')
ax.fill_between(Lz_grid, -10, l_m_min_Omega_r, fc='gray', alpha=0.2)
ax.fill_between(Lz_grid, -10, l_m_min_J_r, fc='gray', alpha=0.2)


resonance_labels = ['CR', 'OLR', '1:1']

for i in range(3):
    ax.axhline(l_grid[i]/m_grid[i], ls=ls[i], color=colors[i])
    try:
        ax.text(3470, l_grid[i]/m_grid[i]-0.25, resonance_labels[i], c=colors[i], fontsize=18, ha='right')
    except:
        continue

ax.axhline(-1/2, ls=':', c='k')
ax.text(3470, -3/4, 'ILR', c='k', fontsize=18, ha='right')


ax.set_xlabel('$J_\phi$ [kpc km/s]', fontsize=16)
ax.set_ylabel('$l/m$', fontsize=16)

ax.text(-3000, -0.73, 'Resonances\nforbidden', color='k', fontsize=18)


ax.set_xlim(min(Lz_grid), max(Lz_grid))
ax.set_ylim(-1, 1.5)

ax.tick_params(top=True, bottom=True, right=True, left=True, direction='in')

plt.show()
plt.close()

#%%

'''Calculate orbits in configuration space'''

from scipy.interpolate import interp1d

def calc_res_orbit(L, J_phi, l, m, phi_peri=0, N_orbits=2, gridsize=10001):
    E = H_res(L, J_phi, l, m)
    J_r = calc_J_r(E, L)
    Omega_r = calc_Omega_r(J_r, L)
    Omega_phi = calc_Omega_phi(J_r, L, J_phi)
    
    
    if (np.isnan(Omega_r) == False) & (np.isinf(Omega_r) == False):
        # Generate series of points uniformly in eccentric anomaly eta
        N = gridsize # Number of points
        N_r_period = N_orbits  # Number of radial periods
        
        
        # Calculate theta_r from uniform grid in eta
        eta = np.linspace(0, 1, N+1) * 2*np.pi*N_r_period
        c = GM/(-2*E) - b
        e = (1 - L**2/(GM*c) * (1 + b/c))**0.5
        
        theta_r = eta - e*c/(c + b) * np.sin(eta)
        
        # Function to calculate eta from theta_r
        try:
            calc_eta = interp1d(theta_r, eta, kind='cubic')
        
        except:
            calc_eta = lambda theta_r: np.zeros(len(theta_r))
        
        # Now recalculate eta from uniform grid of theta_r
        theta_r = np.linspace(0, 1-1/N, N) * 2*np.pi*N_r_period
        eta = calc_eta(theta_r)
        eta_over_pi = eta / np.pi
        
        # Calculate radius
        s = 2 + c/b * (1 - e*np.cos(eta))
        r = b*np.sqrt((s-1)**2 - 1)
        
        # Calculate theta and phi
        theta_phi = Omega_phi/Omega_r * theta_r + phi_peri
        
        phi_shift = 0.5*np.pi*(eta_over_pi - eta_over_pi % 4)*(1 + 1/np.sqrt(1 + 4*GM*b/L**2))
        
        phi = theta_phi - Omega_phi/Omega_r * theta_r + np.sign(J_phi) * (np.arctan2(np.sqrt((1+e)/(1-e)) * np.sin(eta/2), np.cos(eta/2))%(2*np.pi) + 1/np.sqrt(1 + 4*GM*b/L**2) * (np.arctan2(np.sqrt((1+e+2*b/c)/(1-e+2*b/c)) * np.sin(eta/2), np.cos(eta/2))%(2*np.pi))) + phi_shift

        
        # Calculate time and bar angle
        t = theta_r / Omega_r
        phi_b = Omega_b*t
        
        X = r * np.cos(phi-phi_b)
        Y = r * np.sin(phi-phi_b)
        
        X_sun = r0 * np.cos(bar_angle)
        Y_sun = r0 * np.sin(bar_angle)
        
        dist = np.sqrt((X-X_sun)**2 + (Y-Y_sun)**2)
        
        vr = vr_res(r, L, J_phi, l, m, theta_r)
    
    
        return t, X, Y, r, vr, phi, theta_r, theta_phi
    
    else:
        return np.zeros(gridsize), np.zeros(gridsize), np.zeros(gridsize), np.zeros(gridsize), np.zeros(gridsize), np.zeros(gridsize), np.zeros(gridsize), np.zeros(gridsize)

#%%

'''Plot low-Lz orbits as illustration of prograde-retrograde transition'''

L_z_grid = [200, -200, 200]
l_grid = [0, 2, 2]
m = 2
phi_peri_grid = [0, np.pi, np.pi]

colors = ['tab:blue', 'tab:green', 'tab:green']

fig, axs = plt.subplots(3, 2, figsize=(6,8), sharex='row', sharey='col')
plt.subplots_adjust(wspace=0.4, hspace=0)

for i in range(len(l_grid)):
    N = 100000
    N_orbits = 18
    L = abs(L_z_grid[i])
    t, X, Y, r, vr, phi, theta_r, theta_phi = calc_res_orbit(L, L_z_grid[i], l_grid[i], m, phi_peri_grid[i], N_orbits=N_orbits, gridsize=N)
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    
    
    if L_z_grid[i] > 0:
        label = '$l/m=$ '+str(fractions.Fraction(l_grid[i], m_grid[i]))+', $J_\phi>0$'

    else:
        label = '$l/m=$ '+str(fractions.Fraction(l_grid[i], m_grid[i]))+', $J_\phi<0$'
    
    axs[i,0].plot(x, y, c=colors[i], label=label, linewidth=2)
    axs[i,1].plot(X[:int(N/N_orbits)], Y[:int(N/N_orbits)], c=colors[i], label=label, linewidth=2)

    axs[i,0].set_ylabel('$y$ [kpc]', fontsize=18)
    axs[i,1].set_ylabel('$Y$ [kpc]', fontsize=18)
    axs[i,1].text(12, 12, label, fontsize=16)


for ax in axs.flat:
    ax.scatter(0,0, marker='x', c='k')
    
    ax.set_xlim(17,-17)
    ax.set_ylim(-17,17)
    
    ax.tick_params(top=True, bottom=True, right=True, left=True, direction='in', labelsize=14)


axs[-1,0].set_xlabel('$x$ [kpc]', fontsize=18)
axs[-1,1].set_xlabel('$X$ [kpc]', fontsize=18)


axs[0,0].set_title('Inertial frame', fontsize=18)
axs[0,1].set_title('Corotating frame', fontsize=18)

plt.show()
plt.close()


#%%

'''Function to calculate G parameter in pendulum equation for planar orbits'''

def calc_G(J_r, J_phi, l, m, Delta_J=1):
    
    L = abs(J_phi)
    
    Omega_r = calc_Omega_r(J_r, L)
    dOmegar_dJr = -3 * Omega_r / (J_r + 0.5*(L + np.sqrt(L**2 + 4*GM*b)))

    alpha = 0.5*(1 + L/np.sqrt(L**2 + 4*GM*b))

    G = dOmegar_dJr * (l + m*alpha*np.sign(J_phi))**2 + 2 * m**2 * alpha*(1-alpha)*(2*alpha-1) * Omega_r/L
    
    return G

#%%

'''Function to calculate Psi in pendulum equation'''

# Set bar scale length
R_b = 1.9

def Phi_bar(R):
    return -R**2 / (R_b + R)**5


def Psi_1(J_phi, l, m):
    
    t, X, Y, r, vr, phi, theta_r, theta_phi = calc_res_orbit(abs(J_phi), J_phi, l, m, 0, gridsize=100001)
    
    integrand = Phi_bar(r) * np.cos(2*(phi-theta_phi) - l*theta_r)
    
    try:
        index_max = np.where(theta_r > np.pi)[0][0]
    
        Psi_1 = 1/(2*np.pi) * np.trapz(integrand[:index_max], theta_r[:index_max])
        
        return Psi_1
    
    except:
        return 0


J_phi_grid = np.linspace(-800, 800, 201)
l_grid = [0, 1, 2, 3, 4]
m = 2


def calc_G_Psi_1(l_grid, J_phi_grid):
    Psi_1_array = np.zeros((len(l_grid), len(J_phi_grid)))
    G_array = np.zeros((len(l_grid), len(J_phi_grid)))
    
    for i in range(len(l_grid)):
        
        L = abs(J_phi_grid)
        J_r_res = calc_J_r(H_res(L, J_phi_grid, l_grid[i], m), L)
    
        G_array[i] = calc_G(J_r_res, J_phi_grid, l_grid[i], m)
        
        
        for j in range(len(J_phi_grid)):
            Psi_1_array[i,j] = Psi_1(J_phi_grid[j], l_grid[i], m)
    
    return G_array, Psi_1_array


G_array, Psi_1_array = calc_G_Psi_1(l_grid, J_phi_grid)


#%%

'''Plot G*Psi_1 as functions of J_phi'''

fig, ax = plt.subplots(1, figsize=(4,3), sharex=True)
plt.subplots_adjust(hspace=0)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']
ls = ['-', '--','-.', ':', '-']

for i in range(len(l_grid)):
    G_Psi_1 = G_array[i]*Psi_1_array[i]

    label = '$l/m=$ '+str(fractions.Fraction(l_grid[i], m))
    
    ax.plot(J_phi_grid, G_array[i]*Psi_1_array[i]*1e5, color=colors[i], ls=ls[i], label=label)

ax.axhline(0, ls='--', c='grey')

ax.set_xlim(min(J_phi_grid), max(J_phi_grid))

ax.set_xlabel('$J_\phi$ [kpc km/s]', fontsize=16)
ax.set_ylabel('$G\Psi_1$ [arbitrary units]', fontsize=16)

ax.legend(ncol=2, fontsize=10, handlelength=1)

ax.tick_params(top=True, bottom=True, right=True, left=True, direction='in')

plt.show()
plt.close()


#%%

'''Find sign of G*Psi_1 and determine stable pericentre'''

def calc_phi_peri(G_array, Psi_1_array):
    sign = np.sign(G_array*Psi_1_array)
    
    # plt.plot(J_phi_grid, sign)
    
    phi_peri = (-sign - 1)*np.pi/4
    
    return phi_peri
    
phi_peri_array = calc_phi_peri(G_array, Psi_1_array)

#%%

'''Define function to calculate orbits'''

# Define function to calculate orbits
def calc_orbits(l_grid, J_phi_grid, phi_peri_array, m=2, gridsize=10001):

    # Create arrays to store positions, velocities, etc.
    # Dimensions are l, J_phi, pericentre
    X_array = np.zeros((len(l_grid), len(J_phi_grid), 2, gridsize))
    Y_array = np.zeros((len(l_grid), len(J_phi_grid), 2, gridsize))
    r_array = np.zeros((len(l_grid), len(J_phi_grid), 2, gridsize))
    vr_array = np.zeros((len(l_grid), len(J_phi_grid), 2, gridsize))
    dist_array = np.zeros((len(l_grid), len(J_phi_grid), 2, gridsize))
    
    
    for i in range(len(l_grid)):
        
        l = l_grid[i]
        # phi_peri = phi_peri_grid[i] 
        
        for j in range(len(J_phi_grid)):
            
            # Calculate orbital parameters
            J_phi = J_phi_grid[j]
           
            if J_phi > 0:
                phi_peri = phi_peri_array[i,j]
           
            else:
                phi_peri = phi_peri_array[i,j]-np.pi
        
            t, X, Y, r, vr, phi, _, _ = calc_res_orbit(abs(J_phi), J_phi, l, m, phi_peri, gridsize=gridsize)
            
            X_sun = r0 * np.cos(bar_angle)
            Y_sun = r0 * np.sin(bar_angle)
            
            dist_0 = np.sqrt((X-X_sun)**2 + (Y-Y_sun)**2)
            dist_1 = np.sqrt((-X-X_sun)**2 + (-Y-Y_sun)**2)
        
    
            X_array[i,j] = np.array([X, -X])
            Y_array[i,j] = np.array([Y, -Y])
            r_array[i,j] = np.array([r, r])
            vr_array[i,j] = np.array([vr, vr])
            dist_array[i,j] = np.array([dist_0, dist_1])
    
    
    return X_array, Y_array, r_array, vr_array, dist_array

#%%

'''Calculate orbits in chosen resonances for a range of J_phi'''

J_phi_grid = np.linspace(-500, 500, 201)
l_grid = [0, 1, 2, 3, 4]

G_array, Psi_1_array = calc_G_Psi_1(l_grid, J_phi_grid)
phi_peri_array = calc_phi_peri(G_array, Psi_1_array)

X_array, Y_array, r_array, vr_array, dist_array = calc_orbits(l_grid, J_phi_grid, phi_peri_array)
    
#%%

'''Plot orbits'''

fig, axs = plt.subplots(4, len(l_grid), figsize=(10,11), sharex='row', sharey='row')
plt.subplots_adjust(hspace=0.4, wspace=0.)

alpha = 0.04

# Choose maximum distance to plot in (r, v_r) space
dist_max = 8

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']

for i in range(len(l_grid)):
        
    l = l_grid[i]
    m = 2# m_grid[i]
    
    title = '$l/m=$ '+str(fractions.Fraction(l, m))
    axs[0,i].set_title(title, fontsize=20)
        
    for j in range(len(J_phi_grid)):
        J_phi = J_phi_grid[j]
        
        if J_phi > 0:
            axs[0,i].plot(X_array[i,j,0], Y_array[i,j,0], c=colors[i], alpha=alpha)
            axs[1,i].plot(r_array[i,j,0], vr_array[i,j,0], c=colors[i], alpha=alpha)
    
        else:         
            axs[2,i].plot(X_array[i,j,0], Y_array[i,j,0], c=colors[i], alpha=alpha)
            axs[3,i].plot(r_array[i,j,0], vr_array[i,j,0], c=colors[i], alpha=alpha)


pro_cut = J_phi_grid > 0

X_pro_array = X_array[:,pro_cut].flatten()
Y_pro_array = Y_array[:,pro_cut].flatten()
r_pro_array = r_array[:,pro_cut].flatten()
vr_pro_array = vr_array[:,pro_cut].flatten()
dist_pro_array = dist_array[:,pro_cut].flatten()

X_ret_array = X_array[:,~pro_cut].flatten()
Y_ret_array = Y_array[:,~pro_cut].flatten()
r_ret_array = r_array[:,~pro_cut].flatten()
vr_ret_array = vr_array[:,~pro_cut].flatten()
dist_ret_array = dist_array[:,~pro_cut].flatten()


X_sun = r0 * np.cos(bar_angle)
Y_sun = r0 * np.sin(bar_angle)

for ax in axs[0,:].flat:
    ax.scatter(X_sun, Y_sun, marker='o', c='w', edgecolor='k', s=100, zorder=0)
    ax.scatter(X_sun, Y_sun, marker='o', c='k', s=5, zorder=0)
    
    ellipse = patches.Ellipse((0,0), 10, 3, fc='gray', alpha=0.5, zorder=0)
    ax.add_patch(ellipse)

    ax.set_xlim(25, -25)
    ax.set_ylim(-25, 25)
    
    ax.set_xlabel('$X$ [kpc]', fontsize=18)
    
for ax in axs[2,:].flat:
    ax.scatter(X_sun, Y_sun, marker='o', c='w', edgecolor='k', s=100, zorder=0)
    ax.scatter(X_sun, Y_sun, marker='o', c='k', s=5, zorder=0)
    
    ellipse = patches.Ellipse((0,0), 10, 3, fc='gray', alpha=0.5, zorder=0)
    ax.add_patch(ellipse)

    ax.set_xlim(25, -25)
    ax.set_ylim(-25, 25)
    
    ax.set_xlabel('$X$ [kpc]', fontsize=18)

for ax in axs[1,:].flat:
    ax.set_xlim(2, 21)
    ax.set_ylim(-450, 450)
    
    ax.set_xlabel('$r$ [kpc]', fontsize=18)

for ax in axs[3,:].flat:
    ax.set_xlim(2, 21)
    ax.set_ylim(-450, 450)
    
    ax.set_xlabel('$r$ [kpc]', fontsize=18)
    

axs[0,0].set_ylabel('$Y$ [kpc]', fontsize=18)
axs[2,0].set_ylabel('$Y$ [kpc]', fontsize=18)

axs[1,0].set_ylabel('$v_r$ [kpc]', fontsize=18)
axs[3,0].set_ylabel('$v_r$ [kpc]', fontsize=18)

ax1 = fig.add_subplot(211, frameon=False)
ax1.set_ylabel('Prograde ($J_\phi>0$)', fontsize=24, labelpad=20)
ax1.yaxis.set_label_position('right')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.tick_params(top=False, bottom=False, left=False, right=False)

ax1 = fig.add_subplot(212, frameon=False)
ax1.set_ylabel('Retrograde ($J_\phi<0$)', fontsize=24, labelpad=20)
ax1.yaxis.set_label_position('right')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.tick_params(top=False, bottom=False, left=False, right=False)


for ax in axs.flat:
    ax.set_rasterization_zorder(20)
    ax.tick_params(top=True, bottom=True, right=True, left=True, direction='in')

plt.show()
plt.close()


#%%

'''Plot radial phase space weighted by selection function'''

fig, axs = plt.subplots(2,2, figsize=(4,4), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0, wspace=0)

r_bins = np.linspace(4,18,1001)
vr_bins = np.linspace(-400,400,1001)

dist_scale = 1.25
weighting = np.exp(-dist_array / dist_scale)
weighting_pro = np.exp(-dist_pro_array / dist_scale)
weighting_ret = np.exp(-dist_ret_array / dist_scale)

for i in range(len(l_grid)):
    l = l_grid[i]
    m = 2
    
    for j in range(len(J_phi_grid)):
        
        # Calculate orbital parameters
        J_phi = J_phi_grid[j]
       
        if J_phi > 0:
            phi_peri = phi_peri_array[i,j]
       
        else:
            phi_peri = phi_peri_array[i,j]-np.pi
    
        t, X, Y, r, vr, phi, _, _ = calc_res_orbit(abs(J_phi), J_phi, l, m, phi_peri)
        
        if J_phi > 0:
            axs[0,0].plot(r, vr, c=colors[i], alpha=alpha)
        
        else:
            axs[1,0].plot(r, vr, c=colors[i], alpha=alpha)

phi_pro_array = np.arctan2(Y_pro_array, X_pro_array)
phi_ret_array = np.arctan2(Y_ret_array, X_ret_array)

cut_pro = (phi_pro_array > -np.pi/3)*(phi_pro_array < 0)
cut_ret = (phi_ret_array > -np.pi/3)*(phi_ret_array < 0)

hist_pro, _, _ = np.histogram2d(r_pro_array, vr_pro_array, bins=(r_bins, vr_bins), weights=weighting_pro)
hist_ret, _, _ = np.histogram2d(r_ret_array, vr_ret_array, bins=(r_bins, vr_bins), weights=weighting_ret)

hist_pro_antisym = (hist_pro - np.flip(hist_pro, axis=1))/(hist_pro + np.flip(hist_pro, axis=1))
hist_ret_antisym = (hist_ret - np.flip(hist_ret, axis=1))/(hist_ret + np.flip(hist_ret, axis=1))

hist_pro_antisym = np.nan_to_num(hist_pro_antisym)
hist_ret_antisym = np.nan_to_num(hist_ret_antisym)


axs[0,1].imshow(hist_pro.T, extent=(r_bins[0], r_bins[-1], vr_bins[0], vr_bins[-1]), origin='lower', aspect='auto', cmap='Greys', norm=mcolors.LogNorm(vmin=1e-4))#, interpolation='bicubic')
axs[1,1].imshow(hist_ret.T, extent=(r_bins[0], r_bins[-1], vr_bins[0], vr_bins[-1]), origin='lower', aspect='auto', cmap='Greys', norm=mcolors.LogNorm(vmin=1e-4))#, interpolation='bicubic')

axs[-1,0].set_xlabel('$r$ [kpc]', fontsize=16)
axs[-1,1].set_xlabel('$r$ [kpc]', fontsize=16)



for ax in axs[:,0].flat:
    ax.set_ylabel('$v_r$ [km/s]', fontsize=18)


for ax in axs.flat:
    ax.set_ylim(-399, 399)
    ax.tick_params(top=True, bottom=True, right=True, left=True, direction='in')
    ax.set_rasterization_zorder(20)


axs[0,1].text(17.5, 280, '$J_\phi>0$', fontsize=16, ha='right')
axs[1,1].text(17.7, 280, '$J_\phi<0$', fontsize=16, ha='right')

plt.show()
plt.close()

#%%

'''Plot v_r as a function of R and phi'''

phi_pro_array = np.arctan2(Y_pro_array, X_pro_array)
phi_ret_array = np.arctan2(Y_ret_array, X_ret_array)

fig, axs = plt.subplots(2, subplot_kw={'projection': 'polar'}, figsize=(4,8))

vmax = np.maximum(np.nanmax(abs(vr_pro_array)), np.nanmax(abs(vr_ret_array)))

scat_pro = axs[0].scatter(phi_pro_array, r_pro_array, s=0.1, alpha=1, c=vr_pro_array, cmap='seismic', vmin=-vmax, vmax=vmax, rasterized=True)
scat_ret = axs[1].scatter(phi_ret_array, r_ret_array, s=0.1, alpha=1, c=vr_ret_array, cmap='seismic', vmin=-vmax, vmax=vmax, rasterized=True)

cb = fig.colorbar(scat_pro, location='bottom', ax=axs, fraction=0.025, pad=0.04)#, fraction=0.046, pad=0.04)
cb.set_label('$v_r$ [km/s]', fontsize=16)

axs[0].set_title('$J_\phi>0$', fontsize=16)
axs[1].set_title('$J_\phi<0$', fontsize=16)


for ax in axs.flat:
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    
    phi_sun = 2*np.pi-np.pi/6
    
    ax.scatter(phi_sun, r0, marker='o', c='w', edgecolor='k', s=100)
    ax.scatter(phi_sun, r0, marker='o', c='k', s=5)
    
    
    # Draw on ellipse to represent bar
    a = 5
    b = 2
    
    theta_el = np.linspace(0, 2*np.pi, 100)
    r_el = a*b / np.sqrt((b*np.cos(theta_el))**2 + (a*np.sin(theta_el))**2)
    
    ax.fill_between(theta_el, r_el, fc='gray', alpha=0.5, zorder=0)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_ylim(0, 18)
    ax.grid(False)

plt.show()
plt.close()
