# Tropical Cyclone Model

This repository packages the model created in for the [EMPIRIC TC Project](https://github.com/EMPIRIC2/EMPIRIC-AI-emulation). This package was created to setup the model to run as a Google Cloud function, and can be installed through `pip` at the following URL: `git+https://github.com/brown-ccv/tc-model.git`.

The model requires 2 static data files to be able to run. 

1. The model weights (`src/tc_model/data/weights.keras`)
2. The input data (`src/tc_model/data/input_data.hdf5`)

The weights file is only 4 MB, and as a result can be included with the repository. The input data is much larger (~100 MB), and has to be hosted separately. Right now, it is being stored intermediately on Google Drive, and downloaded once the `tc_model` package is installed. Because the Cloud Function will be updated to generate the required input data based on user input parameters, it's not critical to develop a better long term solution to this. When this file is no longer needed, `pyproject.toml` should be updated to remove `pipx` as a dependency and the `hatch` build hook.

If the model is ever retrained, the `weights.keras` file can be replaced with the new version. Code changes should not be required, unless the original model code was sufficiently updated so that the `.keras` file can no longer be loaded.

## Getting Started

To get started with the code, create a virtual environment and install the dependencies:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install --editable
```

This will install the package from the `pyproject.toml` in the virual environment. Changes made to the files in `src/tc_model` will impact the virtual environment.

## Editing the Codebase

All files added in `src/tc_model` will be exposed in the packaged version of the model. For example, a new file `tc_model/foo.py` with a function `bar` could be imported like so:

```python
from tc_model.foo import bar
```

The "API" functions (streamlined consumer facing functionality) is contained within `__init__.py`, and is imported from `tc_model`. Adding a new function `bar` here would be imported like so:

```python
from tc_model import bar
```
Any new consumer facing function written should have a test associated with it to ensure it returns the correct shape of data for a given set of inputs.

## Tests

The tests for this package ensure the functions work correctly. They do not test for *accuracy* of the model, only that the inputs and outputs are in the expected formats.

To run the tests, set up a virtual environment and install the package with the `test` configuration:

```bash
(.venv) $ pip install --editable ".[test]"
```

Then, run the tests by running `pytest`. 
