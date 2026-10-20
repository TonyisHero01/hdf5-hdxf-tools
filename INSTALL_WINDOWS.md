# Windows installation and troubleshooting

This guide is adapted from a contributor's `README_INSTALLATION_PROPOSED.md`, which reports a successful smoke test with **Windows and CPython 3.13.9**. The development environment in this update is not a Windows machine; treat the reported version as a contributor-tested configuration, not a new Windows test by the maintainer.

## Download the source

You can use either **GitHub → Code → Download ZIP** (Git not required) or `git clone` if Git is installed. Extract the ZIP to a normal working folder, such as `Downloads` or `Documents`.

The exact folder nesting depends on how the ZIP was created. **Navigate to the directory that actually contains both `requirements.txt` and `hdxf_viewer.py`**. Do not run installation commands from `C:\Windows\System32` or a parent directory that does not contain the files.

In PowerShell, check the current directory and files:

```powershell
Get-Location
Get-ChildItem
```

## Install Python

Install **64-bit CPython 3.13** from <https://www.python.org/downloads/>. If PowerShell does not recognize `python`, ensure Python is on `PATH`, reopen PowerShell, then check:

```powershell
python --version
```

The `py` launcher is optional. It is not available on every system, so these instructions use `python` after checking the version.

## Create an isolated environment

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The prompt should now begin with `(.venv)`. If activation is blocked, allow scripts **only for this PowerShell session**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If activation is not possible, the commands below can instead invoke `.\.venv\Scripts\python.exe` explicitly.

## Install and validate CPU dependencies

**Activate `.venv` first.** Installing before activation may install all packages globally instead of into the virtual environment.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -c "import sys; print('Python:', sys.executable)"
```

The executable path should end in `\.venv\Scripts\python.exe`. `pip check` should report `No broken requirements found.`

A basic import check:

```powershell
python -c "import numpy, h5py, hdf5plugin, blosc2, scipy, pyFAI; from PIL import Image; print('HDXF core imports OK')"
python -c "import tkinter; print('Tkinter OK')"
```

## Launch the desktop Viewer

```powershell
python .\hdxf_viewer.py
```

This command requires a graphical Windows desktop. To inspect the CLI version without opening the window:

```powershell
python .\hdxf_viewer.py --version
```

## Common installation issues

**`ModuleNotFoundError: numpy` after installing packages:** Check the executable used by `python -m pip` and the one starting the Viewer:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Both should point to the same `.venv` environment. To guarantee the install target:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**`Activate.ps1` is blocked:** See the session-scoped execution policy command above.

**`source` is not recognized:** `source .venv/bin/activate` is for Unix shells. In PowerShell use `.\.venv\Scripts\Activate.ps1`.

**`py` or `python3.13` is not recognized:** These are not universal Windows commands. Check `python --version` and use `python -m venv .venv` if it resolves the intended Python installation.

**`requirements.txt` cannot be found:** Use `Get-ChildItem` to confirm the current directory; change into the folder containing it.

**pyFAI version-check failures:** Some packages do not provide a conventional `__version__` field. Use:

```powershell
python -c "import importlib.metadata as m; print(m.version('pyFAI'))"
```

**HDF5 filters fail:** Confirm `hdf5plugin` is installed in the same environment and imported before reading detector pixels:

```powershell
python -c "import hdf5plugin; print('HDF5 filters imported')"
```

**The Viewer cannot start on a remote/headless session:** A graphical desktop with Tk support is required.

**Converting a master with external data files:** Keep the master and all referenced data files accessible. A master may expose `/entry/data/data_000001` as an ExternalLink targeting `/entry/data/data` in a separate file. The converter's `--frame-dataset` option, when needed, refers to the path **as seen from the master**. Use `python hdf5_to_hdxf.py --help` for all supported settings.

## Optional acceleration, versions and further reading

For CUDA 12 installation instructions, see the single **Optional CUDA 12 installation** section in the main [README](README.md). For code, archive and build versions, see [VERSIONS.md](VERSIONS.md).

Leave the environment with `deactivate` when finished.
