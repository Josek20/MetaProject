# ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
# using Gym
# env = GymEnv("MountainCarContinuous-v0")
using PyCall

magym = pyimport("gym")
