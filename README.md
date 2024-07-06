scikit-utils
================================================================================
Utilities for scikit learn

Contents
================================================================================
- [scikit-utils](#scikit-utils)
- [Contents](#contents)
- [Usage](#usage)
- [Installation](#installation)
- [Development](#development)
  - [Testing](#testing)
  - [Documentation](#documentation)


<a name="usage"></a>

Usage
================================================================================

The main entrypoint is the `skutils` CLI program
```sh
skutils run -i data.csv -o my_outputs/
skutils preprocess -i data.csv -o my_outputs/
skutils fit -i my_outputs/ -o my_outputs/
skutils score -i my_outputs/ -o my_outputs/
skutils visualize -i my_outputs/ -o my_outputs/
```

For the full list of command line options, use the help option
```sh
skutils -h
skutils run -h
...
```

<a name="installation"></a>

Installation
================================================================================

**TODO**

<a name="development"></a>

Development
================================================================================

```sh
PKG_PATH="path/to/scikit-utils"
git clone git@github.com:alexarmstrongvi/scikit-utils.git "$PKG_PATH"
pip install --editable "$PKG_PATH[dev]"
```


## Testing
Manually run tests and checks from within the package dir
```bash
# PyTest
cd $PKG_DIR
pytest
# Coverage checker
coverage run -m pytest
coverage report
# Linting and Formatting
flake8
pylint .
isort --check .
# Complexity checker
radon cc . # replace cc with raw, mi, or hal
# Environment tests
tox run-parallel
```

To followup on a specific file
```bash
pytest               path/to/file.py
python -m doctest -v path/to/file.py
coverage report -m   path/to/file.py
flake8               path/to/file.py
pylint               path/to/file.py
isort --check --diff path/to/file.py
radon cc             path/to/file.py
```

When ready to autoformat the code or run all pre-commit modifications
```bash
isort "$PKG_PATH"
pre-commit run --all-files
```

## Documentation

**TODO**
