#!/usr/bin/env python
# coding: utf-8
# %%
import os
from IPython import get_ipython
import numpy as np
import pathlib

from salvus.toolbox import toolbox
from salvus.flow import simple_config as config
import salvus.namespace as sn

import matplotlib.pyplot as plt
from matplotlib import cm

from utils import my_model

SALVUS_FLOW_SITE_NAME = os.environ.get('SITE_NAME', 'eejit')

# %%
# Parameters
c = 3e8                 # speed of light
mu = 1                  #
rho = 1000              # Density, rho = 1000 kg/m**3
nx, ny = 3000, 3000     # Model size
f_max = 10.0e6          # Maximum frequency
r_ring = 450
ns = 1                  # Number of sources
nr = 380                # Number of receivers
e_light = 1             # Relative permittivity of speed of light
e_rock = 6.795          # Relative permittivity of Rock
e_regolith = 2.375      # Relative permittivity of Regolith
dt, dx = 5.0e-10, 1         # Time step, space step
#max_x, max_y = 700, 700     # Model extension
t_max = 9.5e-6


# Import the model - Relative Permittivity values
data = np.fromfile(file="../../vel1_copy.bin", dtype=np.float32, count=-1,
                   sep='', offset=0)

# Empty array
array_em = np.zeros(nx*ny)
array_em = array_em.reshape(nx,ny)

eps_asteroid = data.reshape(nx, ny)                 # Velocity model
rho_asteroid = np.full((nx, ny), rho, dtype=int)    # Density model
mu_asteroid = np.full((nx, ny), 1, dtype=int)       # Magnetic Permeability

for i in range (nx):
    for j in range (nx):
        if eps_asteroid[i][j] >= 0.26 :
            v_radar = eps_asteroid[i][j]*pow(10,9) / np.sqrt(mu * e_light)
            array_em[i][j] = v_radar
        elif eps_asteroid[i][j] < 0.26 and eps_asteroid[i][j] >= 0.19 :
            v_radar = eps_asteroid[i][j]*pow(10,9) / np.sqrt(mu * e_rock)
            array_em[i][j] = v_radar
        else :
            v_radar = eps_asteroid[i][j]*pow(10,9) / np.sqrt(mu * e_regolith)
            array_em[i][j] = v_radar


plt.imshow(np.rot90(array_em, 3),cm.get_cmap('gnuplot',30))
plt.title('Asteroid model')
plt.colorbar(orientation='vertical')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.savefig('EM_model')
plt.show()


# %%
true_model = my_model(vp=array_em, rho=rho_asteroid,
                      nx=nx, nz=ny)

#plt.figure(figsize=(16, 6))
#plt.subplot(121)
#plt.title("Asteroid model")
#true_model.vp.T.plot()
#plt.subplot(122)
#true_model.rho.T.plot()
#plt.suptitle("Asteroid model")
#plt.show()


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
wavelet = sn.simple_config.stf.Ricker(center_frequency=0.5*f_max)

# Sources
srcs = sn.simple_config.source.cartesian.collections.ScalarPoint2DRing(
    source_time_function = wavelet,  x=450.0, y=450.0, radius=r_ring, count=ns, f=1.0)

# Receivers
recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
    x=0, y=0, radius=r_ring, count=nr, fields=["phi"])

p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)

# BOUNDARIES
vp_min = v_radar.min()
absorbing_par = sn.simple_mesh.basic_mesh.AbsorbingBoundaryParameters(
    reference_velocity=vp_min,
    number_of_wavelengths=6,
    reference_frequency=0.5*f_max,
    free_surface=False)

mesh = toolbox.mesh_from_xarray(
    model_order=4,
    data=true_model,
    slowest_velocity='vp',
    maximum_frequency=f_max,
    elements_per_wavelength=2,
    absorbing_boundaries=(absorbing_side_sets, num_absorbing_layers))


# %%
# salvus simulation object
sim = config.simulation.Waveform(mesh=mesh, sources=srcs, receivers=recs)
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=9.5e-6)
wsc.physics.wave_equation.time_step_in_seconds = 1.0e-10

ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=wavelet,
)

p += sn.SimulationConfiguration(
    name="true_model_new_EM",
    elements_per_wavelength=1.5,
    tensor_order=4,
    max_frequency_in_hertz=f_max,
    absorbing_boundaries=absorbing_par,
    model_configuration=sn.ModelConfiguration(
        background_model=None, volume_models="true_model_EM"
    ),
    # Potentially event dependent settings.
    event_configuration=ec,
)


# %%
# Modeling
# Modeling
p.simulations.launch(
    simulation_configuration="true_model_new_EM",
    events=p.events.get_all(),
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=28,
    wall_time_in_seconds_per_job=10000,
    extra_output_configuration={
        "volume_data": {
            "sampling_interval_in_time_steps": 500,
            "fields": ["phi"]
        }
    })

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

# Obtain the Snapshots
p.simulations.get_simulation_output_directory(
    simulation_configuration="true_model_new_EM", event=p.events.list()[0])

# %%
