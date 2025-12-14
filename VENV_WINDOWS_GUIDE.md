# Activating Virtual Environment on Windows

## The Problem

On Windows, the path `/venv/bin/activate` is for Linux/Mac. Windows uses a different path.

## Correct Commands for Windows

### Option 1: PowerShell (Recommended)
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Option 2: Command Prompt (CMD)
```cmd
venv\Scripts\activate.bat
```

### Option 3: Using the Batch File
Just double-click: `activate_venv.bat`

## Quick Reference

| System | Command |
|--------|---------|
| **Windows (CMD)** | `venv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `.\venv\Scripts\Activate.ps1` |
| **Linux/Mac** | `source venv/bin/activate` |

## If Virtual Environment Doesn't Exist

Create it first:
```bash
python -m venv venv
```

Then activate using the commands above.

## Verify It's Activated

After activation, you should see `(venv)` in your prompt:
```
(venv) C:\Users\rishi\aitxhackathon-2>
```

## Deactivate

When done, deactivate:
```bash
deactivate
```

## Common Issues

### "Execution Policy" Error in PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Cannot find activate"
- Make sure you're in the project root directory
- Check that `venv\Scripts\` exists
- Try creating a new venv: `python -m venv venv`

### "Python not found"
- Make sure Python is installed
- Check it's in your PATH
- Try `python --version` first

