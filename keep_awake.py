"""
keep_awake.py
-------------
Prevents Windows from entering sleep, hibernate, or away mode idle states
while dataset generation is in progress.
"""

import sys
import time
import ctypes

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ES_CONTINUOUS        = 0x80000000
ES_SYSTEM_REQUIRED   = 0x00000001
ES_DISPLAY_REQUIRED  = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040

def keep_awake():
    print("[No-Sleep Mode] Enabled system execution locks (ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED | ES_CONTINUOUS).")
    print("Windows will NOT go to sleep or hibernate while dataset generation runs.")

    while True:
        try:
            # Re-apply thread execution state every 30 seconds to prevent OS idle timeouts
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED | ES_DISPLAY_REQUIRED
            )
        except Exception as err:
            print(f"Warning setting execution state: {err}")
        time.sleep(30)

if __name__ == "__main__":
    keep_awake()
