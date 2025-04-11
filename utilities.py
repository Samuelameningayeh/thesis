import os
import shutil
import subprocess
import urllib

def custom_install_cmdstan(cmdstanpy_version= "1.2.5",
                           tgz_file="colab-cmdstan-2.36.0.tar.gz",
                           tgz_url="https://github.com/stan-dev/cmdstan/releases/download/v2.36.0/colab-cmdstan-2.36.0.tgz"):
    print(f"Installing cmdstanpy=={cmdstanpy_version}")
    subprocess.run(
        ["pip", "install", "--upgrade", f"cmdstanpy=={cmdstanpy_version}"],
        check=True)

    if not os.path.exists(tgz_file):
        print(f"Downloading {tgz_file}")
        urllib.request.urlretrieve(tgz_url, tgz_file)
    else:
        print(f"{tgz_file} has already been downloaded")
    print(f"Unpacking {tgz_file}")
    shutil.unpack_archive(tgz_file)

    print("Installing required libtbb2 Linux package")
    subprocess.run(["apt", "install", "-y", "libtbb2"], check=True)

    print("Setting cmdstan path")
    cmdstan_path_var = "./cmdstan-2.36.0"
    os.environ["CMDSTAN"] = cmdstan_path_var

    # Check that cmdstan can be found
    from cmdstanpy import cmdstan_path

    out = cmdstan_path()

    print("CmdStan successfully installed!")

def test_cmdstan_installation():
    # Run CmdStanPy Hello, World! example
    # Since this script installs cmdstanpy, imports are within each function
    from cmdstanpy import cmdstan_path, CmdStanModel

    bernoulli_stan = os.path.join(cmdstan_path(),
                                  'examples', 'bernoulli', 'bernoulli.stan')

    with open(bernoulli_stan, 'r') as fd:
        print('\n'.join(fd.read().splitlines()))

    bernoulli_data = os.path.join(
        cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')
    with open(bernoulli_data, 'r') as fd:
        print('\n'.join(fd.read().splitlines()))

    # Compile example model bernoulli.stan
    bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

    # Condition on example data bernoulli.data.json
    bern_fit = bernoulli_model.sample(data=bernoulli_data, seed=123)

    # Print a summary of the posterior sample
    bern_fit.summary()
