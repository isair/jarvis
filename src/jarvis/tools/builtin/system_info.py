import datetime
import platform

def get_system_status() -> str:
    """Returns basic system information and the current time."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os_name = platform.system()
    python_version = platform.python_version()
    
    return (
        f"--- Jarvis System Status ---\n"
        f"- Current Time: {now}\n"
        f"- Operating system: {os_name}\n"
        f"- Python Version: {python_version}"
    )