import os
import pathlib


def test_dynamics():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_dynamics")
        os.system(command)


def test_generators():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_generators")
        os.system(command)


def test_mcmc():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_mcmc")
        os.system(command)


def test_priors():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_priors")
        os.system(command)


def test_prorposers():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_proposers")
        os.system(command)


def test_randomgraph():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_randomgraph")
        os.system(command)


def test_utility():
    path_to_tests = pathlib.Path("_midynet/build/tests")
    if (path_to_tests).exists():
        command = "./" + str(path_to_tests / "test_priors")
        os.system(command)
