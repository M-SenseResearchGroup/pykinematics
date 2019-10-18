.. pykinematics usage

=====================
Usage Examples
=====================

Basic Use: MIMU Algorithms
--------------------------
At the core of ``pykinematics`` for processing MIMU data is the *MimuAngles* class, which contains all the sub-methods required to
calibrate and estimate hip joint angles bilaterally. Using the default parameters, estimating joint angles is then as simple as:

.. code-block:: python

    import pykinematics as pk

    static_calibration_data, joint_center_task_data, trial_data = dummy_import_data()

    mimu_estimator = pk.MimuAngles()  # initialize the estimator

    # calibrate based on a static standing trial and a trial for computing the joint center locations
    mimu_estimator.calibrate(static_calibration_data, joint_center_task_data)

    # estimate bilateral hip joint angles
    left_hip_angles, right_hip_angles = mimu_estimator.estimate(trial_data)

Sample Data Example
-------------------
An example script is provided under the github repository to run the
sample data (`download sample data <https://www.uvm.edu/~rsmcginn/download/sample_data.h5>`_).

.. code-block:: python

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    import pykinematics as pk

    plt.style.use('ggplot')


    def import_sample_data(path_to_sample_data):
      # setup the dictionaries for storing the data
      # static calibration: shank data useful to create scaling factor for acceleration to match gravity
      stat_data = {'Lumbar': {}, 'Left thigh': {}, 'Right thigh': {}, 'Left shank': {}, 'Right shank': {}}
      # star calibration needs the shank data as well, for computation of the joint centers
      star_data = {'Lumbar': {}, 'Left thigh': {}, 'Right thigh': {}, 'Left shank': {}, 'Right shank': {}}
      walk_data = {'Lumbar': {}, 'Left thigh': {}, 'Right thigh': {}}

      tasks = ['Static Calibration', 'Star Calibration', 'Treadmill Walk Fast']
      signals = [('Acceleration', 'Accelerometer'), ('Angular velocity', 'Gyroscope'),
                 ('Magnetic field', 'Magnetometer'), ('Time', 'Time')]
      with h5py.File(path_to_sample_data, 'r') as file:
          for task, dat in zip(tasks, [stat_data, star_data, walk_data]):
              for loc in dat.keys():
                  for sigs in signals:
                      dat[loc][sigs[0]] = np.zeros(file[task][loc.title()][sigs[1]].shape)
                      file[task][loc.title()][sigs[1]].read_direct(dat[loc][sigs[0]])
      for dat in [stat_data, star_data, walk_data]:
          for loc in dat.keys():
              dat[loc]['Time'] = dat[loc]['Time'] / 1e6  # convert timestamps to seconds

      return stat_data, star_data, walk_data


    # import the raw sample data
    sample_data_path = 'W:\\Study Data\\Healthy Subjects\\sample_data.h5'
    static_calibration_data, star_calibration_data, walk_fast_data = import_sample_data(sample_data_path)

    # define some additional keyword arguments for optimizations and orientation estimation
    filt_vals = {'Angular acceleration': (2, 12)}

    ka_kwargs = {'opt_kwargs': {'method': 'trf', 'loss': 'arctan'}}
    jc_kwargs = dict(method='SAC', mask_input=True, min_samples=1500, opt_kwargs=dict(loss='arctan'),
                     mask_data='gyr')
    orient_kwargs = dict(error_factor=5e-8, c=0.003, N=64, sigma_g=1e-3, sigma_a=6e-3)

    mimu_estimator = pk.MimuAngles(gravity_value=9.8404, filter_values=filt_vals,
                                  joint_center_kwargs=jc_kwargs, orientation_kwargs=orient_kwargs,
                                  knee_axis_kwargs=ka_kwargs)

    # calibrate the estimator based on Static and Star Calibration tasks
    mimu_estimator.calibrate(static_calibration_data, star_calibration_data)

    # compute the hip joint angles for the Fast Walking on a treadmill
    left_hip_angles, right_hip_angles = mimu_estimator.estimate(walk_fast_data, return_orientation=False)

    # PLOTTING
    fl, axl = plt.subplots(3, sharex=True)
    fr, axr = plt.subplots(3, sharex=True)
    label = [r'Flexion/Extension', 'Ad/Abduction', 'Internal/External Rotation']
    for i in range(3):
      axl[i].plot(left_hip_angles[:, i])
      axr[i].plot(right_hip_angles[:, i])
      axl[i].set_title(label[i])
      axr[i].set_title(label[i])
      axl[i].set_ylabel('Angle [deg]')
      axr[i].set_ylabel('Angle [deg]')

    axl[2].set_xlabel('Sample')
    axr[2].set_xlabel('Sample')
    fl.suptitle('Left Hip Angles')
    fr.suptitle('Right Hip Angles')

    fl.tight_layout(rect=[0, 0.03, 1, 0.95])
    fr.tight_layout(rect=[0, 0.03, 1, 0.95])
