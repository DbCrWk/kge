from kge import Config, Dataset
import uuid

from kge.util.misc import get_git_revision_short_hash
import os
import socket


class Job:

    # static hooks for all Jobs
    # signature: job
    job_created_hooks = []

    def __init__(self, config: Config, dataset: Dataset, parent_job=None):
        self.config = config
        self.dataset = dataset
        self.job_id = str(uuid.uuid4())
        self.parent_job = parent_job
        self.resumed_from_job = None
        userhome = os.path.expanduser("~")
        username = os.path.split(userhome)[-1]
        self.trace_entry = self.trace(
            git_head=get_git_revision_short_hash(),
            username=username,
            hostname=socket.gethostname(),
            folder=config.folder,
        )

        # prepend log entries with the job id. Since we use random job IDs but
        # want short log entries, we only output the first 8 bytes here
        self.config.log_prefix = "[" + self.job_id[0:8] + "] "

        if self.__class__ == Job:
            for f in Job.job_created_hooks:
                f(self)

    def resume(self):
        """Restores all relevant state to resume a previous job.

        Should set `resumed_from_job` to the job ID of the previous job.

        To run the restored job, use :func:`run`.

        """
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def create(config, dataset, parent_job=None):
        """Creates a job for a given configuration."""

        from kge.job import TrainingJob, EvaluationJob, SearchJob

        if config.get("job.type") == "train":
            job = TrainingJob.create(config, dataset, parent_job)
        elif config.get("job.type") == "search":
            job = SearchJob.create(config, dataset, parent_job)
        elif config.get("job.type") == "eval":
            job = EvaluationJob.create(config, dataset, parent_job)
        else:
            raise ValueError("unknown job type")

        # save the configs of all jobs we created
        job.config.save(
            os.path.join(job.config.folder, "config/{}.yaml".format(job.job_id[0:8]))
        )
        return job

    def trace(self, **kwargs):
        """Write a set of key-value pairs to the trace file and automatically append
        information about this job. See `Config.trace` for more information."""
        if self.parent_job is not None:
            kwargs["parent_job_id"] = self.parent_job.job_id
        if self.resumed_from_job is not None:
            kwargs["resumed_from_job_id"] = self.resumed_from_job

        return self.config.trace(
            job_id=self.job_id, job=self.config.get("job.type"), **kwargs
        )
