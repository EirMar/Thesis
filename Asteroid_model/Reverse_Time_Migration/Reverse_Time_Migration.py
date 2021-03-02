#!/usr/bin/env python
# coding: utf-8
# %%
import os
import matplotlib.pyplot as plt
from IPython import get_ipython
import numpy as np
import pathlib
import time

from salvus.toolbox import toolbox
from salvus.flow import simple_config as config
from salvus.opt import smoothing
import salvus.namespace as sn

import utils as ut
from models import my_model
SALVUS_FLOW_SITE_NAME = os.environ.get('SITE_NAME', 'eejit')

# %%
# Parameters
rho = 1000              # Density, rho = 1000 kg/m**3
nx, ny = 3000, 3000     # Model size
dt, dx = 0.02, 1        # Time step, space step
f_max = 20              # Maximum frequency

# Load model
data = np.fromfile(file="../../vel1_copy.bin", dtype=np.float32, count=-1,
                   sep='', offset=0)

vp_asteroid = data.reshape(nx, ny)                  # Velocity model
rho_asteroid = np.full((nx, ny), rho, dtype=int)    # Density model

true_model = my_model(vp=vp_asteroid, rho=rho_asteroid, nx=nx, nz=ny)
true_model.vp.T.plot()


# ------------------------------------------------------------------------------
# SMOOTH VELOCITY MODEL
# ------------------------------------------------------------------------------
#
#
#

# %%
# ------------------------------------------------------------------------------
# CREATE NEW SALVUS PROJECT
# ------------------------------------------------------------------------------
get_ipython().system('rm -rf project_salvus')
vm = sn.model.volume.cartesian.GenericModel(
    name="true_model", data=true_model)
p = sn.Project.from_volume_model(
    path="project_salvus", volume_model=vm)

# stf
wavelet = sn.simple_config.stf.Ricker(center_frequency=10.0)

# Sources
srcs = sn.simple_config.source.cartesian.ScalarPoint2D(
    source_time_function=wavelet, x=-100.0, y=3500.0, f=1)

# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
    x=0, y=0, radius=3500, count=380, fields=["phi"])

p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)

# Waveform Simulation Configuration
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=6.0)

# Event configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0))

# Simulation Configuration - Model - true_model
p += sn.SimulationConfiguration(
    name="target_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz=20.0,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models=["true_model"]),
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)))

# Constan velocity model
homo_mesh = p.simulations.get_mesh("target_model")
homo_mesh.element_nodal_fields["VP"].fill(vp_asteroid[0, 0])
homo_mesh.element_nodal_fields["RHO"].fill(rho_asteroid[0, 0])
p.add_to_project(
    sn.UnstructuredMeshSimulationConfiguration(
        name="direct_wave_sim",
        unstructured_mesh=homo_mesh,
        event_configuration=ec
    )
)
# %%
# ------------------------------------------------------------------------------
# RUN FORWARD SIMULATION
# ------------------------------------------------------------------------------
for sim, store in zip(["target_model", "direct_wave_sim"], [True, False]):
    p.simulations.launch(
        simulation_configuration=sim,
        events=p.events.get_all(),
        site_name=SALVUS_FLOW_SITE_NAME,
        ranks_per_job=48,
        verbosity=0,
        store_adjoint_checkpoints=True,
        wall_time_in_seconds_per_job=1,)

# %%
for sim in ["smooth_model", "target_model"]:
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
        simulation_configuration="smooth_model",
        misfit_configuration="migration",
        store_checkpoints=False,
        events=p.events.list(),
        ranks_per_job=48,
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
    simulation_configuration="smooth_model",
    misfit_configuration="migration",
    events=p.events.list(),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=48,
    wall_time_in_seconds_per_job=1,
    verbosity=True,)

# %%
p.simulations.query(
    simulation_configuration="smooth_model",
    events=p.events.list(),
    misfit_configuration="migration",
    ping_interval_in_seconds=1.0,
    verbosity=2,
    block=True,)

# %%
gradient = p.actions.inversion.sum_gradients(
    simulation_configuration="smooth_model",
    misfit_configuration="migration",
    events=p.events.list(),)

# gradient.write_h5("tot_gradient/gradient.h5")
gradient

# %%
# ------------------------------------------------------------------------------
# COLLECT WAVEFORM DATA
# ------------------------------------------------------------------------------
# true_data = p.waveforms.get(
#    data_name="initial_model", events=p.events.get_all())

# direct_wave = p.waveforms.get(
#    data_name="target_model", events=p.events.get_all())

# Rt = ut.get_gather(true_data)
# Rd = ut.get_gather(direct_wave)
# R = Rt - Rd

# ------------------------------------------------------------------------------
# PLOT SHOT GATHER
# ------------------------------------------------------------------------------
# Normalize and plot the shotgather.
# p_min, p_max = 0.001 * Rt.min(), 0.001 * Rt.max()
# ext = [500, 3500, 3, 0]

# f, ax = plt.subplots(1, 3, figsize=(10, 8))
# ax[0].imshow(Rt[:, :, 10], vmin=p_min, vmax=p_max, extent=ext,
#             aspect="auto", cmap="gray")
# ax[1].imshow(Rd[:, :, 10], vmin=p_min, vmax=p_max, extent=ext,
#             aspect="auto", cmap="gray")
# ax[2].imshow(R[:, :, 10], vmin=p_min, vmax=p_max, extent=ext,
#             aspect="auto", cmap="gray")
# ax[0].set_title("P")
# ax[1].set_title("P direct")
# ax[2].set_title("P - direct wave removed")

# %%

synthetic_data = p.waveforms.get(
    data_name="smooth_model", events=p.events.get_all()
)

synthetic_data[0].plot(component="A", receiver_field="phi")

p.viz.nb.waveforms(
    ["target_model", "smooth_model"], receiver_field="phi"
)


# %%

# # %%
# # ----------------------------------------------------------------------------
# # SMOOTH VELOCITY MODEL
# # ----------------------------------------------------------------------------
# # Boundaries Conditions
# num_absorbing_layers = 10
# absorbing_side_sets = ["x0", "x1", "y0", "y1"]
#
# # Create a mesh from xarray Dataset
# mesh = toolbox.mesh_from_xarray(
#     model_order=4,
#     data=true_model,
#     slowest_velocity="vp",
#     maximum_frequency=0.5*f_max,
#     elements_per_wavelength=1.5,
#     absorbing_boundaries=(absorbing_side_sets, num_absorbing_layers))
#
# # Smooth the model
# mesh.write_h5("true_model.h5")
# smoothing_config = sn.ModelDependentSmoothing(
#     smoothing_lengths_in_wavelengths={
#         "VP": 2.0,
#         "RHO": 2.0
#     },
#     reference_frequency_in_hertz=20,
#     reference_model="true_model.h5",
#     reference_velocities={"VP": "VP", "RHO": "RHO"},
# ).get_smoothing_config()
#
# smooth_model = smoothing.run(
#     site_name="eejit",
#     ranks_per_job=48,
#     model=mesh,
#     smoothing_config=smoothing_config,
#     wall_time_in_seconds_per_job=10000
# )
# smooth_model
