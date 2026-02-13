# Process Safety & Shutdown Design

## Design Principle: Self-Only Shutdown

**CRITICAL**: This codebase must NEVER use process hunting or external termination.

### ✅ ALLOWED: Self-Only Shutdown

The bot can only stop **itself**, using:
- `sys.exit(0)` - Clean exit with Python cleanup
- Raising exceptions that propagate to main
- Setting `self.running = False` flags
- File-based signaling (e.g., `/tmp/astra_kill`)

### ❌ PROHIBITED: External Process Termination

**NEVER use these patterns:**
- `pkill` / `killall` / `taskkill` - Process name matching
- `os.kill(pid, signal.SIGKILL)` - Direct PID termination
- `subprocess.run(["pkill", "python"])` - Shell process hunting
- `psutil.Process(pid).kill()` - External process termination
- Any code that searches for processes by name and terminates them

### Why This Matters

**Problem**: Process hunting is dangerous:
- Can kill unrelated processes with same name
- Can interfere with system services
- Can cause data loss in other applications
- Violates principle of isolation

**Solution**: Self-only shutdown:
- Process only controls its own lifecycle
- Clean exit allows proper cleanup (file closing, WS shutdown, etc.)
- No risk of affecting other processes
- Respects OS process isolation

### Current Implementation

**Kill Switch**: `/tmp/astra_kill` file-based signaling
```python
# CORRECT: Self-only shutdown via file check
KILL_SWITCH_PATH = Path("/tmp/astra_kill")
if KILL_SWITCH_PATH.exists():
    sys.exit(0)  # Stop ONLY this process
```

**Graceful Shutdown**: Internal flag-based
```python
# CORRECT: Internal state flag
self.running = False  # Signal internal tasks to stop
await asyncio.sleep(1)  # Allow tasks to wind down
```

### Code Review Checklist

Before merging any PR, verify:
- [ ] No `pkill` / `killall` / `taskkill` commands
- [ ] No `os.kill()` calls
- [ ] No `signal.SIGKILL` / `signal.SIGTERM` to external PIDs
- [ ] No `psutil` process hunting
- [ ] No subprocess calls that terminate other processes
- [ ] All shutdown logic is self-only (`sys.exit()`, internal flags)

### Verification Command

Run this to check for dangerous patterns:
```bash
rg -n "pkill|killall|os\.kill|signal\.SIGKILL|SIGTERM|subprocess.*kill|taskkill|psutil.*kill" . --type py
```

If this returns ANY results (except this documentation), **reject the change**.

### Alternative: Process Group Shutdown (If Needed)

If you need to stop child processes (e.g., background tasks), use **process groups**:

```python
# CORRECT: Start in own process group
import os
import signal

# On startup
os.setpgrp()  # Create new process group

# On shutdown (stops all children in this group)
os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
```

This is safe because:
- Only affects processes in **this process's group**
- No process name matching
- Isolated from system processes

### Summary

✅ **Self-only shutdown**: `sys.exit()`, internal flags, file signaling
❌ **Process hunting**: `pkill`, `os.kill()`, name matching
🔒 **Principle**: A process can only stop itself or its direct children via process groups

**This design is non-negotiable for system safety.**
