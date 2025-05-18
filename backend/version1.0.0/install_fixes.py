import os
import shutil
import sys
from pathlib import Path
import subprocess
import platform

def copy_fix_files():
    """Copy fix files to the main backend directory."""
    try:
        # Get paths
        script_dir = Path(__file__).parent
        backend_dir = script_dir.parent
        
        # Files to copy
        files_to_copy = ['documorph_fixes.py']
        
        # Copy files
        print(f"Copying fix files to {backend_dir}...")
        for file in files_to_copy:
            src = script_dir / file
            dst = backend_dir / file
            if src.exists():
                shutil.copy2(src, dst)
                print(f"✅ Copied {file} to {dst}")
            else:
                print(f"❌ Error: Source file {src} not found")
        
        # Install required dependencies
        print("\nInstalling required dependencies...")
        try:
            # Determine the correct pip command based on platform and environment
            pip_cmd = "pip"
            if platform.system() == "Windows":
                if os.path.exists(sys.prefix + "\\Scripts\\pip.exe"):
                    pip_cmd = f"{sys.prefix}\\Scripts\\pip.exe"
                elif os.path.exists(sys.prefix + "\\Scripts\\pip3.exe"):
                    pip_cmd = f"{sys.prefix}\\Scripts\\pip3.exe"
            else:
                if shutil.which("pip3"):
                    pip_cmd = "pip3"
            
            # Install dependencies
            packages = [
                "pymupdf",
                "pandas",
                "numpy",
                "pillow",
                "PyPDF2",
                "pdf2image",
                "python-docx"
            ]
            
            for package in packages:
                print(f"Installing {package}...")
                result = subprocess.run([pip_cmd, "install", "--upgrade", package], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Successfully installed {package}")
                else:
                    print(f"⚠️ Warning: Could not install {package}: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not install dependencies: {str(e)}")
                
        print("\nInstallation complete!")
        print("\nTo apply the fixes:")
        print("1. Make sure DocuMorph AI is not running")
        print("2. Run: python -m streamlit run backend/version1.0.0/integrate_fixes.py")
        print("3. Click 'Apply Fixes' in the web interface")
        print("4. Restart DocuMorph AI")
        
        return True
    except Exception as e:
        print(f"❌ Error during installation: {str(e)}")
        return False

if __name__ == "__main__":
    copy_fix_files() 