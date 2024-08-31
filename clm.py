import pyopencl as cl

def check_opencl():
    platforms = cl.get_platforms()
    if len(platforms) == 0:
        print("No OpenCL platforms found")
        return
    
    for platform in platforms:
        print(f"Platform: {platform.name}")
        devices = platform.get_devices()
        for device in devices:
            print(f"  Device: {device.name}")
            print(f"    Type: {cl.device_type.to_string(device.type)}")
            print(f"    Vendor: {device.vendor}")
            print(f"    Version: {device.version}")
            print(f"    Max Compute Units: {device.max_compute_units}")

if __name__ == "__main__":
    check_opencl()