#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:

import os
import matplotlib.pyplot as plt
from IPython import get_ipython
import xarray as xr
import numpy as np
import time

from salvus.mesh.unstructured_mesh_utils import extract_model_to_regular_grid
import salvus.namespace as sn
from models import my_model
import utils as ut

SALVUS_FLOW_SITE_NAME = os.environ.get('SITE_NAME', 'eejit')


# Parameters
ns = 1                  # Number of sources
nr = 380                # Number of receivers
r_ring = 450           # Satellite altitud
t_max = 9.5e-5             # Simulation time
rho = 1000              # Density, rho = 1000 kg/m**3
nx, ny = 3000, 3000     # Model size
dt, dx = 0.1, 1        # Time step, space step
max_x, max_y = 500, 500   # Model extension
c = 3e8                 # speed of light
mu = 1                  #
f_max = 15.0e6          # Maximum frequency

# Load model
data = np.fromfile(file="../../vel1_copy.bin", dtype=np.float32, count=-1,
                   sep='', offset=0)

eps_asteroid = data.reshape(nx, ny)                 # Velocity model
rho_asteroid = np.full((nx, ny), rho, dtype=int)    # Density model
mu_asteroid = np.full((nx, ny), 1, dtype=int)       # Magnetic Permeability
v_radar = c / (mu * np.sqrt(eps_asteroid))          # Radar


true_model = my_model(vp=v_radar, rho=rho_asteroid,
                      max_x=max_x, max_y=max_y)
#true_model.vp.T.plot()


# ------------------------------------------------------------------------------
# CREATE NEW SALVUS PROJECT
# ------------------------------------------------------------------------------
get_ipython().system('rm -rf project_salvus')
vm = sn.model.volume.cartesian.GenericModel(
    name="true_model", data=true_model)
p = sn.Project.from_volume_model(
    path="project_salvus", volume_model=vm)

# stf
wavelet = sn.simple_config.stf.Ricker(center_frequency=0.5*f_max)

# Sources
srcs = sn.simple_config.source.cartesian.collections.ScalarPoint2DRing(
    x=0, y=0, radius=r_ring, count=ns, f=1.0)

# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
    x=0, y=0, radius=r_ring, count=nr, fields=["phi"])

# BOUNDARIES
vp_min = v_radar.min()
absorbing_par = sn.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    reference_velocity=vp_min,
    number_of_wavelengths=6,
    reference_frequency=0.5*f_max,
    free_surface=False)

p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)

# Waveform Simulation Configuration
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=t_max)
#wsc.physics.wave_equation.time_step_in_seconds = dt

# Event configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=wavelet)

# Simulation Configuration - Model - true_model
p += sn.SimulationConfiguration(
    name="RTM_sim",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz=f_max,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models=["true_model"]),
    absorbing_boundaries=absorbing_par,
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=wavelet))

# Constan velocity model
homo_mesh = p.simulations.get_mesh("RTM_sim")
homo_mesh.element_nodal_fields["VP"].fill(v_radar[0, 0])
homo_mesh.element_nodal_fields["RHO"].fill(rho_asteroid[0, 0])
p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        name="direct_wave_sim",
        unstructured_mesh=homo_mesh,
        event_configuration=ec))

# SMOOTHED VELOCITY MODEL
smooth_mesh = p.actions.inversion.smooth_model(
    model=p.simulations.get_mesh("RTM_sim"),
    smoothing_configuration=sn.ConstantSmoothing(
        smoothing_lengths_in_meters={"VP": 20.0,
                                     "RHO": 20.0, }),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=48,
    verbosity=2,
    wall_time_in_seconds_per_job=10000,)

p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        name="RTM_smooth_sim",
        unstructured_mesh=smooth_mesh,
        event_configuration=ec))

# ------------------------------------------------------------------------------
# RUN FORWARD SIMULATION
# ------------------------------------------------------------------------------
for sim, store in zip(["RTM_smooth_sim", "direct_wave_sim"], [True, False]):
    p.simulations.launch(
        simulation_configuration=sim,
        events=p.events.get_all(),
        site_name=SALVUS_FLOW_SITE_NAME,
        ranks_per_job=48,
        verbosity=2,
        store_adjoint_checkpoints=True,
        wall_time_in_seconds_per_job=10000,)

for sim in ["RTM_smooth_sim", "direct_wave_sim"]:
    p.simulations.query(
        simulation_configuration=sim,
        events=p.events.list(),
        verbosity=2,
        block=True,)

# ------------------------------------------------------------------------------
# Compute adjoint sources and gradients
# ------------------------------------------------------------------------------
def misfit_func(data_synthetic: np.ndarray,
                data_observed: np.ndarray,
                sampling_rate_in_hertz: float):
    adj_src = data_synthetic - data_observed
    return 1.0, adj_src


p += sn.MisfitConfiguration(
    name="migration",
    observed_data="direct_wave_sim",
    misfit_function=misfit_func,
    receiver_field="phi")

# Computed misfits before running adjoint simulation
misfits = None
while not misfits:
    misfits = p.actions.inversion.compute_misfits(
        simulation_configuration="RTM_smooth_sim",
        misfit_configuration="migration",
        store_checkpoints=False,
        events=p.events.list(),
        ranks_per_job=48,
        site_name=SALVUS_FLOW_SITE_NAME,
        wall_time_in_seconds_per_job=10000,
        verbosity=2,
    )
    time.sleep(5.0)

print(misfits)

# ------------------------------------------------------------------------------
# RUN ADJOINT SIMULATION
# ------------------------------------------------------------------------------
p.simulations.launch_adjoint(
    simulation_configuration="RTM_smooth_sim",
    misfit_configuration="migration",
    events=p.events.list(),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=48,
    wall_time_in_seconds_per_job=10000,
    verbosity=True,)

p.simulations.query(
    simulation_configuration="RTM_smooth_sim",
    events=p.events.list(),
    misfit_configuration="migration",
    ping_interval_in_seconds=1.0,
    verbosity=2,
    block=True,)

p.viz.nb.gradients(
    simulation_configuration="RTM_smooth_sim",
    misfit_configuration="migration",
    events=p.events.list(),)

gradient = p.actions.inversion.sum_gradients(
    simulation_configuration="RTM_smooth_sim",
    misfit_configuration="migration",
    events=p.events.list(),)

# gradient.write_h5("tot_gradient/gradient.h5")
gradient

# ------------------------------------------------------------------------------
# COLLECT WAVEFORM DATA
# ------------------------------------------------------------------------------
fwd_data = p.waveforms.get(
    data_name="RTM_smooth_sim", events=p.events.get_all())
direct_wave = p.waveforms.get(
    data_name="direct_wave_sim", events=p.events.get_all())

gather_full  = ut.get_gather(data=fwd_data, rcv_field="phi")
gather_direct = ut.get_gather(data=direct_wave, rcv_field="phi")
gather = gather_full - gather_direct     # Direct wave removal

# ------------------------------------------------------------------------------
# PLOT SHOT GATHER
# ------------------------------------------------------------------------------
# Normalize and plot the shotgather.
p_min, p_max = 0.01 * gather_full.min(), 0.01 * gather_full.max()
ext = [0, 360, t_max, 0]
ns = 15
theta_lim = [0, 360]

f, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(gather_full[:, :, ns], vmin=p_min, vmax=p_max, extent=ext,
             aspect="auto", cmap="gray")
ax[1].imshow(gather_direct[:, :, ns], vmin=p_min, vmax=p_max, extent=ext,
             aspect="auto", cmap="gray")
ax[2].imshow(gather[:, :, ns], vmin=0.5*p_min, vmax=0.5*p_max, extent=ext,
             aspect="auto", cmap="gray")

ax[0].set_xlim(theta_lim)
ax[1].set_xlim(theta_lim)
ax[2].set_xlim(theta_lim)

ax[0].set_title("P")
ax[1].set_title("P direct")
ax[2].set_title("P - direct wave removed")
plt.savefig("shot.pdf")

ds = extract_model_to_regular_grid(
    mesh=gradient,
    ds=xr.Dataset(
        coords={"x": np.linspace(-max_x, +max_y, nx),
                "y": np.linspace(-max_x, +max_y, ny), }),
    pars=["RHO"],
    verbose=True,
)

RHO = ds.RHO.data
ny_cut = 800
nx_cut = 300
RHO = RHO[nx_cut:nx-nx_cut, ny_cut:ny-ny_cut]

p_min, p_max = 0.1*RHO.min(), 0.1*RHO.max()
ext = [-max_x, max_x, -max_y, max_y, ]
x_lim, y_lim = [-4000, 3500], [-3800, 3000]

plt.figure(figsize=(8, 6))
plt.imshow(RHO.T, vmin=p_min, vmax=p_max, extent=ext,
           aspect="auto", cmap="gray")
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.savefig("plot.pdf")
