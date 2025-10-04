# Developmet information


# Project setup

```bash
# conda environment setup
conda create --name d2g_evaluation_env python=3.13
conda activate d2g_evaluation_env
```

```bash
# install only dependencies
uv pip install -r pyproject.toml --all-extras

# check installed packages
uv pip list
```

```bash
# pre-commit
pre-commit --version
pre-commit install
pre-commit run --all-files
```

```bash
#pytest
pytest --version
pytest -v
pytest -vv
coverage run -m pytest
coverage report
coverage html
```

```bash
#mypy
mypy .
```