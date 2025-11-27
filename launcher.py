import sys
import os
import platform

# Đảm bảo Python tìm thấy modules
sys.path.append(os.getcwd())

# Import script gốc
try:
    import spark_jobs.clean_data_spark as original_script
except ImportError:
    # Fallback xử lý đường dẫn
    sys.path.append(os.path.dirname(os.getcwd()))
    import spark_jobs.clean_data_spark as original_script

# Hàm giả để thay thế hàm setup Windows


def dummy_setup(base_dir):
    print(f"🐧 Detected {platform.system()}. Bypassing Windows Setup.")
    return


# LOGIC MONKEY PATCHING
if platform.system() != "Windows":
    print(f"⚙️ Applying cross-platform patch for {platform.system()}...")
    # Ghi đè hàm setup_windows_env bằng hàm rỗng
    original_script.setup_windows_env = dummy_setup
else:
    print("🪟 Windows detected. Using original configuration.")

if __name__ == "__main__":
    original_script.run()
