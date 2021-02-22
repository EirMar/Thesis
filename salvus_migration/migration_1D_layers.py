# %%
################################################################################
# FORWARD MODELLING USING SALVUS - SPECTRAL ELEMENT SOLVER
#    Salvus 0.11.19
#    Author: D. Vargas
################################################################################
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import salvus.namespace as sn

import utils as ut
SALVUS_FLOW_SITE_NAME = os.environ.get("SITE_NAME", "eejit")

x_min, x_max = 0.0, 4000.0
y_min, y_max = 0.0, 2000.0
z0 = 50

# Sources/Receivers coordinates
s0, ns, ds = x_min+500, 201, 150              # Sources
r0, nr, dr = x_min+500, 201, 150              # Receivers
s = [np.arange(s0, s0 + ns*ds, ds), (y_max-z0)*np.ones(ns)]  # [sx, sy]
r = [np.arange(r0, r0 + nr*dr, dr), (y_max-z0)*np.ones(nr)]  # [rx, ry]
max_frequency = 40.0   # Max freq to resolve - elements/wavelength

# Model
vp = [0.0, 2110.0, 2179.0, 2022.0, 1918.0, 2385.0, 1760.0, 2259.0]
rho = [2000.0 for i in range(len(vp))]
depth = [0.0, 500.0, 700.0, 900.0, 1250.0, 1500.0, 1800.0, 2000.0]
layered_model = ut.get_layers(
    vp=vp, rho=rho, depth=depth, x_max=4000.0, nsmooth=5)

# ------------------------------------------------------------------------------
# CREATE NEW SALVUS PROJECT
# ------------------------------------------------------------------------------
# !rm -rf salvus_project
# This line should not be here!
vm = sn.model.volume.cartesian.GenericModel(
    name="layered_model", data=layered_model)
p = sn.Project.from_volume_model(path="salvus_project", volume_model=vm)

# %%
# ------------------------------------------------------------------------------
# SET: SOURCES, RECEIVERS, BOUNDARIES, SIMULATION CONFIGURATION
# ------------------------------------------------------------------------------
# SOURCES @ Surface
sources = [sn.simple_config.source.cartesian.ScalarPoint2D(
    x=x_i,
    y=y_i,
    f=1,
    source_time_function=None)
    for i, (x_i, y_i) in enumerate(zip(s[0], s[1]))]

# RECEIVERS @ Surface
receivers = [sn.simple_config.receiver.cartesian.Point2D(
    x=x_i,
    y=y_i,
    station_code=f"{i:03d}",
    fields=["phi_t"])
    for i, (x_i, y_i) in enumerate(zip(r[0], r[1]))]

# BOUNDARIES
vp_min = 1760
absorbing_par = sn.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    reference_velocity=vp_min,
    number_of_wavelengths=6,
    reference_frequency=0.5*max_frequency,
    free_surface=False)

# Add event collection to the project ------------------------
for _i, src in enumerate(sources):
    p += sn.EventCollection.from_sources(
        sources=[src], receivers=receivers, event_name_starting_index=_i)

# Waveform simulation configuration
dt, tmax, t_sam = 0.0005, 3.0, 30
wsc = sn.WaveformSimulationConfiguration(
    start_time_in_seconds=0.0,
    end_time_in_seconds=tmax,
    time_step_in_seconds=dt)
# wsc.physics.wave_equation.boundaries = [absorbing]

# Event configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=0.5*max_frequency,
                                        time_shift_in_seconds=0.1))
# Background model
bm = sn.model.background.homogeneous.IsotropicAcoustic(vp=vp[1], rho=2000.0)

# Simulation Configuration
p += sn.SimulationConfiguration(
    name="initial_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz=max_frequency,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models="layered_model"),
    absorbing_boundaries=absorbing_par,
    event_configuration=ec)

p += sn.SimulationConfiguration(
    name="target_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz=max_frequency,
    model_configuration=sn.ModelConfiguration(background_model=bm),
    absorbing_boundaries=absorbing_par,
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=0.5*max_frequency,
                                            time_shift_in_seconds=0.1)))

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
        ranks_per_job=40,
        verbosity=2,
        store_adjoint_checkpoints=store,
        wall_time_in_seconds_per_job=300,)

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
    receiver_field="phi_t")

# Computed misfits before running adjoint simulation
misfits = None
while not misfits:
    misfits = p.actions.inversion.compute_misfits(
        simulation_configuration="initial_model",
        misfit_configuration="migration",
        store_checkpoints=False,
        events=p.events.list(),
        ranks_per_job=40,
        site_name=SALVUS_FLOW_SITE_NAME,
        wall_time_in_seconds_per_job=300,
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
    ranks_per_job=40,
    wall_time_in_seconds_per_job=300,
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
