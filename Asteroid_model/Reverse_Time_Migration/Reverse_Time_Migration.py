#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


import os
import numpy as np
import time
import xarray as xr
from pathlib import Path
import pathlib
import salvus.namespace as sn
from salvus.flow import simple_config as config
import salvus.mesh.unstructured_mesh as um
import salvus.mesh.structured_grid_2D as sg2d
from salvus.flow import simple_config as config
from salvus.toolbox import toolbox
import salvus.namespace as sn
import matplotlib.pyplot as plt

# Variable used in the notebook to determine which site
# is used to run the simulations.

#SALVUS_FLOW_SITE_NAME=os.environ.get('SITE_NAME','local')
SALVUS_FLOW_SITE_NAME=os.environ.get('eejit','eejit')



file = "vel1_copy.bin"
dt = np.dtype([('time', '<u2'),('time1', '<u2'),('time2', np.float32),('time3', np.float32)])
data = np.fromfile(file, dtype=np.float32, count=-1, sep='', offset=0)

my_array_rel_perm=data.reshape(3000,3000)
my_array_rho=np.full((3000,3000),1000, dtype=int)


def my_model():
    nx, nz = 3000, 3000
    x = np.linspace(-4000, +4000, nx)
    y = np.linspace(-4000, +4000, nx)
    xx, yy = np.meshgrid(x, y, indexing = "ij")

    #put the array elements into the appropriate part of the model xarray structure
    ds = xr.Dataset ( data_vars= {"vp": (["x", "y"], my_array_rel_perm),
     "rho": (["x", "y"], my_array_rho),},
     coords={"x": x, "y": y},)

    #Transform velocity to SI units (m/s).
    ds['vp'] *=10000

    return ds

true_model = my_model()


# ------------------------------------------------------------------------------
# CREATE NEW SALVUS PROJECT
# ------------------------------------------------------------------------------
!rm -rf salvus_project
vm = sn.model.volume.cartesian.GenericModel(
    name="true_model", data=true_model)
p = sn.Project.from_volume_model(path="salvus_project_master_thesis", volume_model=vm)


wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)
mesh_frequency =  wavelet.center_frequency


# Sources
srcs = sn.simple_config.source.cartesian.ScalarPoint2D(
    source_time_function=wavelet, x=-100.0, y=3500.0, f=1)


# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
        x=0, y=0, radius=3500, count=380, fields=["phi"]
)

p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)

# Waveform Simulation Configuration
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=6.0)

# Event configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0))

# Simulation Configuration
p += sn.SimulationConfiguration(
    name="initial_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz = 20.0,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models="true_model"),
    event_configuration=ec)

p += sn.SimulationConfiguration(
    name="target_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz = 20.0,
    model_configuration=sn.ModelConfiguration(background_model="true_model"),
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)))

p.simulations.get_mesh("initial_model")


# %%
# ------------------------------------------------------------------------------
# RUN FORWARD SIMULATION
# ------------------------------------------------------------------------------
for sim, store in zip(["initial_model", "target_model"], [True, False]):
    p.simulations.launch(
        simulation_configuration=sim,
        events=p.events.get_all(),
        site_name=SALVUS_FLOW_SITE_NAME,
        ranks_per_job=48,
        verbosity=0,
        store_adjoint_checkpoints=store,
        wall_time_in_seconds_per_job=1,)

# %%
for sim in ["initial_model", "target_model"]:
    p.simulations.query(
        simulation_configuration=sim,
        events=p.events.list(),
        verbosity=2,
        block=True,)

# %%
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
    observed_data="target_model",
    misfit_function=misfit_func,
    receiver_field="phi")

# Computed misfits before running adjoint simulation
misfits = None
while not misfits:
    misfits = p.actions.inversion.compute_misfits(
        simulation_configuration="initial_model",
        misfit_configuration="migration",
        store_checkpoints=False,
        events=p.events.list(),
        ranks_per_job=4,
        site_name=SALVUS_FLOW_SITE_NAME,
        wall_time_in_seconds_per_job=1,
        verbosity=2,
    )
    time.sleep(5.0)

print(misfits)

# %%
# ------------------------------------------------------------------------------
# RUN ADJOINT SIMULATION
# ------------------------------------------------------------------------------
p.simulations.launch_adjoint(
    simulation_configuration="initial_model",
    misfit_configuration="migration",
    events=p.events.list(),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=4,
    wall_time_in_seconds_per_job=1,
    verbosity=True,)

# %%
p.simulations.query(
    simulation_configuration="initial_model",
    events=p.events.list(),
    misfit_configuration="migration",
    ping_interval_in_seconds=1.0,
    verbosity=2,
    block=True,)

# %%
gradient = p.actions.inversion.sum_gradients(
    simulation_configuration="initial_model",
    misfit_configuration="migration",
    events=p.events.list(),)

# gradient.write_h5("tot_gradient/gradient.h5")
gradient

# %%
# ------------------------------------------------------------------------------
# COLLECT WAVEFORM DATA
# ------------------------------------------------------------------------------
true_data = p.waveforms.get(
    data_name="initial_model", events=p.events.get_all())

direct_wave = p.waveforms.get(
    data_name="target_model", events=p.events.get_all())

Rt = ut.get_gather(true_data)
Rd = ut.get_gather(direct_wave)
R = Rt - Rd

# ------------------------------------------------------------------------------
# PLOT SHOT GATHER
# ------------------------------------------------------------------------------
# Normalize and plot the shotgather.
p_min, p_max = 0.001 * Rt.min(), 0.001 * Rt.max()
ext = [500, 3500, 3, 0]

f, ax = plt.subplots(1, 3, figsize=(10, 8))
ax[0].imshow(Rt[:, :, 10], vmin=p_min, vmax=p_max, extent=ext,
             aspect="auto", cmap="gray")
ax[1].imshow(Rd[:, :, 10], vmin=p_min, vmax=p_max, extent=ext,
             aspect="auto", cmap="gray")
ax[2].imshow(R[:, :, 10], vmin=p_min, vmax=p_max, extent=ext,
             aspect="auto", cmap="gray")
ax[0].set_title("P")
ax[1].set_title("P direct")
ax[2].set_title("P - direct wave removed")

# %%

# %%
