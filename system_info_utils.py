import platform
import subprocess
import psutil
import re # For parsing command outputs
import json # For structured error messages or complex outputs

def run_command(command):
    """Helper function to run a shell command and return its output."""
    try:
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        # print(f"Command '{command}' failed with error: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        # print(f"Command not found for: {command}")
        return None

def get_os_info():
    """Gets basic OS information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "node": platform.node(),
    }

def get_cpu_details():
    """Gets CPU model, core counts, and frequencies."""
    details = {
        "model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }
    try:
        freq = psutil.cpu_freq()
        details["current_frequency_mhz"] = freq.current
        details["min_frequency_mhz"] = freq.min
        details["max_frequency_mhz"] = freq.max
    except Exception: # psutil.cpu_freq() might not be available or might fail
        details["frequency_info"] = "Not available"
    return details

def get_detailed_ram_info():
    """Gets detailed RAM information (total, available, used, free, percent)."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "free_gb": round(mem.free / (1024**3), 2),
        "percent_used": mem.percent,
    }

def get_detailed_storage_info():
    """Lists disk partitions and their usage."""
    partitions_info = []
    try:
        partitions = psutil.disk_partitions()
        for p in partitions:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                partitions_info.append({
                    "device": p.device,
                    "mountpoint": p.mountpoint,
                    "fstype": p.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "percent_used": usage.percent,
                })
            except Exception: # Can fail for some special partitions like /dev/loop
                partitions_info.append({
                    "device": p.device,
                    "mountpoint": p.mountpoint,
                    "fstype": p.fstype,
                    "usage_info": "Could not retrieve usage (possibly a pseudo-filesystem or access denied)",
                })
        return partitions_info
    except Exception as e:
        return {"error": f"Could not retrieve disk partitions: {str(e)}"}


def get_gpu_info():
    """Gets GPU model(s) and VRAM if possible."""
    gpus = []
    system = platform.system()

    if system == "Linux":
        # Try lspci for GPU models
        lspci_output = run_command("lspci -mmv | grep -A10 -E 'VGA compatible controller|3D controller'")
        if lspci_output:
            devices = lspci_output.strip().split("\n\n")
            for device_info in devices:
                gpu = {"model": "Unknown", "vram_mb": "N/A", "vendor": "Unknown"}
                model_match = re.search(r"Device:\s*(.*)", device_info)
                if model_match:
                    gpu["model"] = model_match.group(1).strip()

                vendor_match = re.search(r"Vendor:\s*(.*)", device_info)
                if vendor_match:
                    gpu["vendor"] = vendor_match.group(1).strip()

                # If NVIDIA, try nvidia-smi for VRAM
                if "nvidia" in gpu["model"].lower() or "nvidia" in gpu["vendor"].lower():
                    nvidia_smi_output = run_command("nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader,nounits")
                    if nvidia_smi_output:
                        # Assuming single NVIDIA GPU for simplicity here, or matching based on lspci output
                        # This part might need refinement for multi-NVIDIA GPU systems to match correctly
                        lines = nvidia_smi_output.splitlines()
                        if lines: # Take the first GPU if multiple are reported by nvidia-smi
                            parts = lines[0].split(',')
                            gpu["model"] = parts[0].strip() # nvidia-smi often has a more precise model name
                            gpu["vram_mb"] = int(parts[1].strip())
                gpus.append(gpu)
        if not gpus: # Fallback if lspci parsing fails or no clear device
            gpus.append({"model": "No identifiable GPU found via lspci", "vram_mb": "N/A"})

    elif system == "Darwin": # macOS
        sp_display_output = run_command("system_profiler SPDisplaysDataType -json")
        if sp_display_output:
            try:
                display_data = json.loads(sp_display_output)
                controllers = display_data.get("SPDisplaysDataType", [])
                for controller in controllers:
                    gpu = {"model": "Unknown", "vram_mb": "N/A", "vendor": "Unknown"}
                    # Model name can be under sGPU_Chipset_Model or spdisplays_vram
                    model_name = controller.get("sppci_model", controller.get("spdisplays_display_type", "Unknown GPU"))
                    if model_name == "Unknown GPU" and "spdisplays_chipset-model" in controller : # Intel integrated often here
                         model_name = controller.get("spdisplays_chipset-model")

                    gpu["model"] = model_name

                    vram = controller.get("spdisplays_vram", "N/A") # e.g. "4 GB" or "1536 MB"
                    if vram != "N/A":
                        # Attempt to parse VRAM into MB
                        num_part = vram.split(" ")[0]
                        unit_part = vram.split(" ")[1].upper() if len(vram.split(" ")) > 1 else ""
                        try:
                            num = int(num_part)
                            if unit_part == "GB":
                                gpu["vram_mb"] = num * 1024
                            elif unit_part == "MB":
                                gpu["vram_mb"] = num
                            else:
                                gpu["vram_mb"] = vram # Store as string if parsing fails
                        except ValueError:
                            gpu["vram_mb"] = vram # Store original string if number conversion fails

                    # Vendor info might be part of the model name or sppci_vendor_id
                    vendor_id = controller.get("sppci_vendor_id", "")
                    if "nvidia" in vendor_id.lower() or "nvidia" in gpu["model"].lower(): gpu["vendor"] = "NVIDIA"
                    elif "amd" in vendor_id.lower() or "amd" in gpu["model"].lower() or "ati" in gpu["model"].lower(): gpu["vendor"] = "AMD"
                    elif "intel" in vendor_id.lower() or "intel" in gpu["model"].lower(): gpu["vendor"] = "Intel"

                    gpus.append(gpu)
            except json.JSONDecodeError:
                gpus.append({"model": "Failed to parse system_profiler JSON output", "vram_mb": "N/A"})
        if not gpus:
             gpus.append({"model": "No identifiable GPU found via system_profiler", "vram_mb": "N/A"})
    else:
        gpus.append({"model": "GPU information not available for this OS", "vram_mb": "N/A"})
    return gpus


def get_system_model_info():
    """Gets system manufacturer and product name."""
    system = platform.system()
    info = {"manufacturer": "Unknown", "model": "Unknown"}

    if system == "Linux":
        manufacturer = run_command("cat /sys/devices/virtual/dmi/id/sys_vendor")
        model = run_command("cat /sys/devices/virtual/dmi/id/product_name")
        # Fallback to dmidecode if direct sysfs fails or for more detail (though dmidecode needs sudo often)
        if not manufacturer or not model :
            if run_command("command -v dmidecode"): # Check if dmidecode exists
                 manufacturer = run_command("sudo dmidecode -s system-manufacturer")
                 model = run_command("sudo dmidecode -s system-product-name")
            else:
                manufacturer = manufacturer or "N/A (dmidecode not found)"
                model = model or "N/A (dmidecode not found)"

        info["manufacturer"] = manufacturer if manufacturer else "N/A"
        info["model"] = model if model else "N/A"

    elif system == "Darwin": # macOS
        sp_hardware_output = run_command("system_profiler SPHardwareDataType -json")
        if sp_hardware_output:
            try:
                hw_data = json.loads(sp_hardware_output)
                # SPHardwareDataType is an array with one element
                if hw_data.get("SPHardwareDataType"):
                    hw_info = hw_data["SPHardwareDataType"][0]
                    info["manufacturer"] = "Apple Inc." # Standard for Macs
                    info["model"] = hw_info.get("machine_name", "Unknown Mac") + " (" + hw_info.get("machine_model", "") + ")"
                    # info["serial_number"] = hw_info.get("serial_number", "N/A") # If needed
            except json.JSONDecodeError:
                 info["model"] = "Failed to parse system_profiler JSON for hardware"
        else:
            info["model"] = "Could not get system_profiler hardware data"

    else:
        info["manufacturer"] = "Not available for this OS"
        info["model"] = "Not available for this OS"
    return info

if __name__ == '__main__':
    print("OS Info:", json.dumps(get_os_info(), indent=2))
    print("\nCPU Details:", json.dumps(get_cpu_details(), indent=2))
    print("\nRAM Info:", json.dumps(get_detailed_ram_info(), indent=2))
    print("\nStorage Info:", json.dumps(get_detailed_storage_info(), indent=2))
    print("\nGPU Info:", json.dumps(get_gpu_info(), indent=2))
    print("\nSystem Model Info:", json.dumps(get_system_model_info(), indent=2))
