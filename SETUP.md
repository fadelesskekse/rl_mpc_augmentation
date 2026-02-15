# Setup Instructions

This project requires additional dependencies that need to be set up before running.

## Required Dependencies

1. **DRCL Assets** (https://github.com/drcl-deploy/assets) - Contains the G1 robot URDF and meshes
2. **Unitree RL Lab** (https://github.com/unitreerobotics/unitree_rl_lab) - Contains rewards and utilities

## Installation Steps

### Step 1: Clone DRCL Assets

```bash
cd ~  # or wherever you want to put it
git clone https://github.com/drcl-deploy/assets.git
cd assets
```

### Step 2: Install DRCL Assets

Install the assets package in Isaac Lab's Python environment:

```bash
cd ~/assets  # or wherever you cloned it
~/IsaacLab/isaaclab.sh -p -m pip install -e .
```

### Step 3: Clone Unitree RL Lab

```bash
cd ~  # or wherever you want to put it
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab
```

### Step 4: Install Unitree RL Lab

```bash
cd ~/unitree_rl_lab  # or wherever you cloned it
~/IsaacLab/isaaclab.sh -p -m pip install -e .
```

### Step 5: Install This Extension

```bash
cd ~/rl_mpc_augmentation/source/rl_mpc_augmentation
~/IsaacLab/isaaclab.sh -p -m pip install -e .
```

## Verify Installation

After installation, verify that all packages are installed:

```bash
~/IsaacLab/isaaclab.sh -p -c "import assets; import unitree_rl_lab; import rl_mpc_augmentation; print('All packages installed successfully!')"
```

## Troubleshooting

### Error: "Asset convert failed with error status: Unsupported Format (/meshes/pelvis)"

This error indicates that:
1. The DRCL assets repository is not installed, OR
2. The mesh files are missing or corrupted

**Solution:**
- Make sure you've cloned the DRCL assets repository completely (including all mesh files)
- Verify the assets package is installed: `~/IsaacLab/isaaclab.sh -p -c "import assets; print(assets.__file__)"`
- Check that the mesh files exist in the assets repository

### Error: "ModuleNotFoundError: No module named 'assets'"

**Solution:**
- Install the DRCL assets package as shown in Step 2 above
- Make sure you're using Isaac Lab's Python environment (via `isaaclab.sh -p`)

### Error: "ModuleNotFoundError: No module named 'unitree_rl_lab'"

**Solution:**
- Install the unitree_rl_lab package as shown in Step 4 above

## Alternative: Add to PYTHONPATH

If you prefer not to install the packages, you can add them to your Python path. However, installing them is recommended.

```bash
export PYTHONPATH=$PYTHONPATH:~/assets:~/unitree_rl_lab
```

But note: You still need to use `isaaclab.sh -p` to run scripts, which may not preserve your PYTHONPATH. Installation is the recommended approach.
