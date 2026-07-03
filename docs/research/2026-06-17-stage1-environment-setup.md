# Stage 1 Environment Setup

Date: 2026-06-17

Scope: configure an isolated project environment for the original HOPE repository. This stage does not run checkpoint evaluation or training.

## Result

Stage 1 environment setup is usable.

- Environment path: `D:\Github\HOPE\.venv`
- Python executable: `D:\Github\HOPE\.venv\Scripts\python.exe`
- Python version: `3.13.12`
- PyTorch: `2.11.0+cu128`
- CUDA visible to PyTorch: yes
- GPU detected by PyTorch: `NVIDIA GeForce RTX 4080 SUPER`
- Dependency consistency: `pip check` reports no broken requirements.
- Original HOPE environment reset: passed with original observation keys and shapes.

## Why Python 3.13 Was Used

The original README recommends Python 3.8 in a Conda environment. This machine did not have Conda or Python 3.8 available on PATH:

```powershell
py -0p
```

Available launcher interpreters were:

```text
-V:3.14 *        C:\Python314\python.exe
-V:3.13          C:\Users\zhang\AppData\Local\Programs\Python\Python313\python.exe
```

Because Python 3.14 is the default and is newer than needed for this ML project, the isolated environment was created with Python 3.13.

## Commands Run

Create the environment:

```powershell
cd D:\Github\HOPE
py -3.13 -m venv .venv
```

Upgrade packaging tools inside the isolated environment:

```powershell
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -m pip install --upgrade pip setuptools wheel
```

Install repository requirements:

```powershell
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

Install PyTorch from the official CUDA 12.8 wheel index:

```powershell
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The PyTorch install adjusted `setuptools` inside `.venv` to `70.2.0` because the torch wheel declares `setuptools<82`.

## Verification Commands

Dependency consistency:

```powershell
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -m pip check
```

Result:

```text
No broken requirements found.
```

Key imports and versions:

```powershell
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -c "import sys, numpy, shapely, pygame, gym, cv2, scipy, torch; print('exe=', sys.executable); print('python=', sys.version.split()[0]); print('numpy=', numpy.__version__); print('shapely=', shapely.__version__); print('pygame=', pygame.version.ver); print('gym=', gym.__version__); print('cv2=', cv2.__version__); print('scipy=', scipy.__version__); print('torch=', torch.__version__)"
```

Result:

```text
exe= D:\Github\HOPE\.venv\Scripts\python.exe
python= 3.13.12
numpy= 2.4.6
shapely= 2.1.2
pygame= 2.6.1
gym= 0.26.2
cv2= 4.13.0
scipy= 1.17.1
torch= 2.11.0+cu128
```

PyTorch CUDA verification:

```powershell
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Result:

```text
torch= 2.11.0+cu128
cuda_available= True
torch_cuda= 12.8
device= NVIDIA GeForce RTX 4080 SUPER
```

Original HOPE environment reset:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' -c "from env.car_parking_base import CarParking; from env.env_wrapper import CarParkingWrapper; e=CarParking(render_mode='rgb_array', verbose=False); w=CarParkingWrapper(e); obs=w.reset(None, None, 'Normal'); print(sorted(obs.keys())); print({k: None if v is None else v.shape for k,v in obs.items()}); e.close()"
```

Result:

```text
initializing action mask
['action_mask', 'img', 'lidar', 'target']
{'img': (3, 64, 64), 'lidar': (120,), 'target': (5,), 'action_mask': (42,)}
```

## Warnings

Importing `gym 0.26.2` prints a deprecation warning:

```text
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
```

This warning did not block imports or environment reset in Stage 1. Do not migrate to Gymnasium during Stage 1 or Stage 2; the current goal is to validate and reproduce the original HOPE code path first.

`gym.spaces.Box` also prints:

```text
WARN: Box bound precision lowered by casting to float64
```

This warning appears during environment construction and did not block reset.

## Issue Encountered

The first environment reset verification used:

```python
w.reset(level='Normal')
```

That failed with:

```text
TypeError: CarParkingWrapper.reset() got an unexpected keyword argument 'level'
```

Root cause: `CarParkingWrapper.reset()` accepts positional `*args` only and forwards them to the wrapped environment. The verification was corrected to:

```python
w.reset(None, None, 'Normal')
```

This matches the original training scripts' positional call pattern.

## Frozen Packages

```text
absl-py==2.4.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
einops==0.8.2
filelock==3.29.0
fonttools==4.63.0
fsspec==2026.4.0
grpcio==1.81.1
gym==0.26.2
gym-notices==0.1.0
HeapDict==1.0.1
Jinja2==3.1.6
kiwisolver==1.5.0
Markdown==3.10.2
MarkupSafe==3.0.3
matplotlib==3.11.0
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
opencv-python==4.13.0.92
packaging==26.2
pillow==12.2.0
protobuf==7.35.1
pygame==2.6.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scipy==1.17.1
setuptools==70.2.0
shapely==2.1.2
six==1.17.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
torch==2.11.0+cu128
tqdm==4.68.3
typing_extensions==4.15.0
Werkzeug==3.1.8
wheel==0.47.0
```

## Next Stage

Stage 2 should run the author-provided checkpoint with the isolated Python executable:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 10 --visualize False
```
