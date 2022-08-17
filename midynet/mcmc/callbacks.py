from _midynet.mcmc import callbacks as cb
from _midynet.mcmc.callbacks import *
from midynet.wrapper import Wrapper


__all__ = (
    "CallBack",
    "CallBack",
    "CollectEdgesOnSweep",
    "CollectGraphOnSweep",
    "CollectPartitionOnSweep",
    "CollectLikelihoodOnSweep",
    "CollectPriorOnSweep",
    "CollectJointOnSweep",
    "CheckConsistencyOnSweep",
    "CheckSafetyOnSweep",
    "VerboseDisplay",
)


def CollectEdgesOnSweep(labeled=False, nested=False):
    if nested and labeled:
        return cb._CollectNestedBlockLabeledEdgeMultiplicityOnSweep()
    elif not nested and labeled:
        return cb._CollectBlockLabeledEdgeMultiplicityOnSweep()
    return cb._CollectEdgeMultiplicityOnSweep()


def CollectGraphOnSweep(labeled=False, nested=False):
    if nested and labeled:
        return cb._CollectNestedBlockLabeledGraphOnSweep()
    elif not nested and labeled:
        return cb._CollectBlockLabeledGraphOnSweep()
    return cb._CollectGraphOnSweep()


def CollectPartitionOnSweep(nested=False, type="reconstruction"):
    if nested and type == "reconstruction":
        return cb._CollectNestedPartitionOnSweepForReconstruction()
    elif nested and type == "community":
        return cb._CollectNestedPartitionOnSweepForCommunity()
    elif not nested and type == "reconstruction":
        return cb._CollectPartitionOnSweepForReconstruction()
    return cb._CollectPartitionOnSweepForCommunity()
