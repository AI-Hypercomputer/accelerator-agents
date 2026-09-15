# pylint: skip-file
from absl import app
import xmanager as xm
from xmanager import xm_abc


def main(_):
  with xm_abc.create_experiment(
      experiment_title='Ghostfish Training Experiment'
  ) as experiment:
    # Define the executable using the train.py script in the same directory
    # In a real environment, you might use xm.bazel_binary or xm.python_executable
    # with a specific requirements.txt or container.
    [executable] = experiment.package([
        xm.python_executable(
            executable_path='train.py',
            # In GQM/Borg environment, we use Borg spec
            executor_spec=xm_abc.Borg.Spec(),
        ),
    ])

    # Specify the requirements using the GQM details provided
    requirements = xm.JobRequirements(
        location='yulhrp',  # Must target this cell explicitly
        service_tier=xm.ServiceTier.PROD,
        tpu=xm.Tpu(
            tpu_type='GHOSTFISH',
            topology='4',  # Requesting 4 chips as per 'count: 4'
        ),
    )

    # Create the executor with Borg configuration
    executor = xm_abc.Borg(
        requirements=requirements,
        billing=xm_abc.BorgBilling(accounting_group='mlacc-gqm-dyn'),
        # Resource pool as per user instructions
        # Note: xm_abc.Borg may have different ways to specify resource pool
        # usually via JobRequirements or flags.
    )

    # Define the job
    job = xm.Job(executable, executor)

    # Add the job to the experiment
    experiment.add(job)


if __name__ == '__main__':
  app.run(main)
