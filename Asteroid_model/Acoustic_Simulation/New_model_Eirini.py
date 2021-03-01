#!/usr/bin/env python
# coding: utf-8
# %%
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pathlib

from salvus.toolbox import toolbox
from salvus.flow import simple_config as config
import salvus.namespace as sn

# Set Salvus site. Where to run the simulations
# SALVUS_FLOW_SITE_NAME=os.environ.get('SITE_NAME','local')
SALVUS_FLOW_SITE_NAME = os.environ.get('SITE_NAME', 'eejit')

# %%
rho = 1000      # Density, rho = 1000 kg/m**3

data = np.fromfile(file="vel1_copy.bin", dtype=np.float32, count=-1,
                   sep='', offset=0)
vp_asteroid = data.reshape(3000, 3000)
# Density model
rho_asteroid = np.full((3000, 3000), rho, dtype=int)
# print(rho_asteroid)


# %%


plt.imshow(np.rot90(vp_asteroid, 3))
plt.title('Asteroid model')
plt.colorbar(orientation='vertical')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()


# %%


def my_model():
    nx, nz = 3000, 3000
    x = np.linspace(-4000, +4000, nx)
    y = np.linspace(-4000, +4000, nx)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # put the array elements into the appropriate part of the model xarray structure
    ds = xr.Dataset(data_vars={"vp": (["x", "y"], vp_asteroid),  "rho": (
        ["x", "y"], rho_asteroid), }, coords={"x": x, "y": y},)

    # Transform velocity to SI units (m/s).
    ds['vp'] *= 10000

    return ds


# %%


# Plot the xarray dataset.

true_model = my_model()

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.title("Asteroid model")
true_model.vp.T.plot()
plt.subplot(122)
true_model.rho.T.plot()
plt.suptitle("Asteroid model")
plt.show()


# #### Stability Test

# %%


# Stability Test

dt = 0.02
dx = 1

eps = vp_asteroid.min() * dt / dx

print('Stability criterion =', eps)


# %%


test_Nyquist = 1 / (2*dt)

print(test_Nyquist)


# %%


# Ricker wavelet

wavelet = config.stf.Ricker(center_frequency=10.0)
f, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(*wavelet.get_stf(), color='purple')
ax[0].set_xlabel("Time (sec)")
ax[0].set_ylabel("Amplitude")

ax[1].plot(*wavelet.get_power_spectrum(), color='gold')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude")

ax[0].grid()
ax[1].grid()

plt.tight_layout()

plt.show()
print(type(wavelet))


# %%


# Create new Project

get_ipython().system('rm -rf project')
if pathlib.Path("project").exists():
    print("Opening existing project.")
    p = sn.Project(path="project")
else:
    print("Creating new project.")
    vm = sn.model.volume.cartesian.GenericModel(
        name="true_model_2", data=true_model
    )
    p = sn.Project.from_volume_model(path="project", volume_model=vm)


# %%


wavelet = sn.simple_config.stf.Ricker(center_frequency=10.0)
mesh_frequency = wavelet.center_frequency


# Sources
srcs = sn.simple_config.source.cartesian.ScalarPoint2D(
    source_time_function=wavelet, x=-100.0, y=3500.0, f=1)


# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
    x=0, y=0, radius=3500, count=380, fields=["phi"]
)


p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)


# %%


# Boundaries Conditions

num_absorbing_layers = 10
absorbing_side_sets = ["x0", "x1", "y0", "y1"]

# Create a mesh from xarray Dataset
mesh = toolbox.mesh_from_xarray(
    model_order=4,
    data=true_model,
    slowest_velocity="vp",
    maximum_frequency=mesh_frequency,
    elements_per_wavelength=1.5,
    absorbing_boundaries=(absorbing_side_sets, num_absorbing_layers))


# %%


# Visualize mesh
mesh


# ## Start Simulation

# %%


sim = config.simulation.Waveform(mesh=mesh, sources=srcs, receivers=recs)


# ## Create Snapshots

# %%


#sim.output.point_data.format = "hdf5"


# %%


# Save the volumetric wavefield for visualization purposes.
#sim.output.volume_data.format = "hdf5"
#sim.output.volume_data.filename = "output.h5"
#sim.output.volume_data.fields = ["phi"]
#sim.output.volume_data.sampling_interval_in_time_steps = 10


# %%


sim.validate()


# %%


# Visualize Simulation

sim


# %%


# Waveform Simulation Configuration
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=6.0)


# %%


# Event Configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0),
)

# Simulation Configuration

p += sn.SimulationConfiguration(
    name="true_model_new",
    elements_per_wavelength=1.5,
    tensor_order=4,
    max_frequency_in_hertz=mesh_frequency,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models="true_model_2"
    ),
    # Potentially event dependent settings.
    event_configuration=ec,
)


# %%


p.simulations.launch(
    simulation_configuration="true_model_new",
    events=p.events.get_all(),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=4,
    wall_time_in_seconds_per_job=1,
    extra_output_configuration={
        "volume_data": {
            "sampling_interval_in_time_steps": 10,
            "fields": ["phi"]
        }
    })


# %%


p.simulations.query(block=True)


# %%


p.simulations.get_mesh("true_model_new")


# %%


true_data = p.waveforms.get(
    data_name="true_model_new", events=p.events.get_all()
)


# %%


p.viz.nb.waveforms(
    data=["true_model_new"],
    receiver_field="phi",
)


# %%


true_data[0].plot(component="A", receiver_field="phi")


# Obtain the Snapshots

p.simulations.get_simulation_output_directory(
    simulation_configuration="true_model_new", event=p.events.list()[0])
