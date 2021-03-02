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

SALVUS_FLOW_SITE_NAME = os.environ.get('SITE_NAME', 'local')

# %%
# Parameters
c = 3e8                 # speed of light
mu = 1                  #
rho = 1000              # Density, rho = 1000 kg/m**3
nx, ny = 3000, 3000     # Model size
f_max = 20              # Maximum frequency

# Import the model - Relative Permittivity values
data = np.fromfile(file="../../vel1_copy.bin", dtype=np.float32, count=-1,
                   sep='', offset=0)

eps_asteroid = data.reshape(nx, ny)                 # Velocity model
rho_asteroid = np.full((nx, ny), rho, dtype=int)    # Density model
mu_asteroid = np.full((nx, ny), 1, dtype=int)       # Magnetic Permeability
v_radar = c / (mu * np.sqrt(eps_asteroid))          # Radar

plt.imshow(np.rot90(v_radar, 3))
plt.title('Asteroid model')
plt.colorbar(orientation='vertical')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()


# %%
true_model = my_model(vp=v_radar, rho=rho_asteroid, nx=nx, nz=ny)

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.title("Asteroid model")
true_model.vp.T.plot()
plt.subplot(122)
true_model.rho.T.plot()
plt.suptitle("Asteroid model")
plt.show()


# %%
get_ipython().system('rm -rf project')
if pathlib.Path("project").exists():
    print("Opening existing project.")
    p = sn.Project(path="project")
else:
    print("Creating new project.")
    vm = sn.model.volume.cartesian.GenericModel(
        name="true_model_EM", data=true_model
    )
    p = sn.Project.from_volume_model(path="project", volume_model=vm)

# stf
wavelet = sn.simple_config.stf.Ricker(center_frequency=15.0e6)

# Sources
srcs = sn.simple_config.source.cartesian.ScalarPoint2D(
    source_time_function=wavelet, x=0.0, y=450.0, f=1)

# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
    x=0, y=0, radius=450, count=380, fields=["phi"])

p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)


# In[13]:


num_absorbing_layers = 10
absorbing_side_sets = ["x0", "x1", "y0", "y1"]

mesh = toolbox.mesh_from_xarray(
    model_order=4,
    data=true_model,
    slowest_velocity='vp',
    maximum_frequency=f_max,
    elements_per_wavelength=2,
    absorbing_boundaries=(absorbing_side_sets, num_absorbing_layers))


# %%


mesh


# %%


sim = config.simulation.Waveform(mesh=mesh, sources=srcs, receivers=recs)


# %%


sim


# %%


#wsc = sn.WaveformSimulationConfiguration(start_time_in_seconds=1e-9)
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=9.5e-6)
wsc.physics.wave_equation.time_step_in_seconds = 1.0e-10


# %%


ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=20.0e6),
)


p += sn.SimulationConfiguration(
    name="true_model_new_EM",
    elements_per_wavelength=1.5,
    tensor_order=4,
    max_frequency_in_hertz=mesh_frequency,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models="true_model_EM"
    ),
    # Potentially event dependent settings.
    event_configuration=ec,
)


# %%


p.simulations.launch(
    simulation_configuration="true_model_new_EM",
    events=p.events.get_all(),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=1,

)


# %%


p.simulations.query(block=True)


# %%


p.simulations.get_mesh("true_model_new_EM")


# %%


true_data = p.waveforms.get(
    data_name="true_model_new_EM", events=p.events.get_all()
)


# %%


true_data[0].plot(component="A", receiver_field="phi")


# %%
