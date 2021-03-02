#!/usr/bin/env python
# coding: utf-8
# %%
import os
import matplotlib.pyplot as plt
from IPython import get_ipython
import numpy as np
import pathlib

from salvus.toolbox import toolbox
from salvus.flow import simple_config as config
import salvus.namespace as sn

from utils import my_model

# Set Salvus site. Where to run the simulations
# SALVUS_FLOW_SITE_NAME=os.environ.get('SITE_NAME','local')
SALVUS_FLOW_SITE_NAME = os.environ.get('SITE_NAME', 'eejit')

# %%
# Parameters
rho = 1000              # Density, rho = 1000 kg/m**3
nx, ny = 3000, 3000     # Model size
dt, dx = 0.02, 1        # Time step, space step
nyquist = 1 / (2*dt)    # Nyquist
f_max = 20              # Maximum frequency

# Load model
data = np.fromfile(file="../../vel1_copy.bin", dtype=np.float32, count=-1,
                   sep='', offset=0)

vp_asteroid = data.reshape(nx, ny)                  # Velocity model
rho_asteroid = np.full((ny, ny), rho, dtype=int)    # Density model

# Stability Test
eps = vp_asteroid.min() * dt / dx
print('Stability criterion = {}'.format(eps))
print('Nyquist = {}'.format(nyquist))

plt.imshow(np.rot90(vp_asteroid, 3))
plt.title('Asteroid model')
plt.colorbar(orientation='vertical')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()


# %%
true_model = my_model(vp=vp_asteroid, rho=rho_asteroid, nx=nx, nz=ny)

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.title("Asteroid model")
true_model.vp.T.plot()
plt.subplot(122)
true_model.rho.T.plot()
plt.suptitle("Asteroid model")
plt.show()


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


# %%
# Create new salvus Project
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

# stf
wavelet = sn.simple_config.stf.Ricker(center_frequency=0.5*f_max)

# Sources
srcs = sn.simple_config.source.cartesian.ScalarPoint2D(
    source_time_function=wavelet, x=-100.0, y=3500.0, f=1)

# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
    x=0, y=0, radius=3500, count=380, fields=["phi"])

p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)

# Boundaries Conditions
num_absorbing_layers = 10
absorbing_side_sets = ["x0", "x1", "y0", "y1"]

# Create a mesh from xarray Dataset
mesh = toolbox.mesh_from_xarray(
    model_order=4,
    data=true_model,
    slowest_velocity="vp",
    maximum_frequency=f_max,
    elements_per_wavelength=1.5,
    absorbing_boundaries=(absorbing_side_sets, num_absorbing_layers))


# %%
# salvus simulation object
sim = config.simulation.Waveform(mesh=mesh, sources=srcs, receivers=recs)
# # Save the volumetric wavefield for visualization purposes.
# sim.output.volume_data.format = "hdf5"
# sim.output.volume_data.filename = "output.h5"
# sim.output.volume_data.fields = ["phi"]
# sim.output.volume_data.sampling_interval_in_time_steps = 10
sim.validate()
mesh        # Visualize mesh


# %%
# Event Configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
        end_time_in_seconds=6.0),
    wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0),
)

# Simulation Configuration
p += sn.SimulationConfiguration(
    name="true_model_new",
    elements_per_wavelength=1.5,
    tensor_order=4,
    max_frequency_in_hertz=f_max,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models="true_model_2"
    ),
    # Potentially event dependent settings.
    event_configuration=ec,
)


# %%
# Modeling
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
