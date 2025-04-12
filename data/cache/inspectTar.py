# tar_inspector.py
import tarfile
import json
import tempfile
import os
from PIL import Image

def inspect_tar_file(tar_path, max_files=3):
    """Inspect contents of a tar file and display sample files"""
    print(f"\nInspecting tar file: {tar_path}")
    
    try:
        with tarfile.open(tar_path, "r") as tar:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as tmpdir:
                print("\nFile structure:")
                members = tar.getmembers()
                
                # Print first 5 files to show structure
                for m in members[:5]:
                    print(f" - {m.name} ({'dir' if m.isdir() else f'{m.size} bytes'})")
                
                # Extract all files
                tar.extractall(path=tmpdir)
                
                # Process extracted files
                processed = 0
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        if processed >= max_files:
                            return
                            
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, tmpdir)
                        
                        print(f"\n=== Contents of {rel_path} ===")
                        
                        try:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                # Display image
                                img = Image.open(full_path)
                                print(f"Image format: {img.format}")
                                print(f"Image size: {img.size}")
                                img.show()  # Opens in default image viewer
                                
                            elif file.lower().endswith('.json'):
                                # Display JSON contents
                                with open(full_path, 'r') as f:
                                    data = json.load(f)
                                    print(json.dumps(data, indent=2))
                                    
                            else:
                                print(f"Unsupported file type: {file}")
                                
                            processed += 1
                            
                        except Exception as e:
                            print(f"Error processing {file}: {str(e)}")
                            
    except Exception as e:
        print(f"Error inspecting tar file: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect contents of SA-1B tar files')
    parser.add_argument('tar_path', type=str, help='Path to the tar file to inspect')
    parser.add_argument('--max_files', type=int, default=3, 
                       help='Maximum number of files to display')
    args = parser.parse_args()
    
    inspect_tar_file(args.tar_path, args.max_files)