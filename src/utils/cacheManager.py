import os
import shutil

def construct_file_name(used_percentage, splitname, extension="arrow", prefix="mapped"):
    return f"{prefix}_{used_percentage}_{splitname}.{extension}"
    
class cacheManager:
    to_cleanup_on_exit: list = []
    to_copy_to_cache: list = []
    def __init__(self, cache_dir, fast_cache_dir=None):
        self.cache_dir = cache_dir
        self.fast_cache_dir = fast_cache_dir

    def get_file_path(self, filename):
        # use filename as directory 
        filename_without_ext = os.path.splitext(filename)[0]
        cache_dir = os.path.join(self.cache_dir, filename_without_ext)
        fast_cache_dir = os.path.join(self.fast_cache_dir, filename_without_ext)
        fast_cache_path = os.path.join(fast_cache_dir, filename)
        if self.fast_cache_dir is not None:
            # check if file exists in cache dir
            os.makedirs(fast_cache_dir, exist_ok=True)
            if os.path.exists(cache_dir):
                # copy to fast cache dir
                print(f"Copying {cache_dir} to {fast_cache_dir} for faster access...")
                shutil.copytree(cache_dir, fast_cache_dir, dirs_exist_ok=True)
            else:
                self.to_copy_to_cache.append((fast_cache_dir, cache_dir))
                print(f"File {cache_dir} does not exist. Will move {fast_cache_dir} to {cache_dir} on exit.")
            self.to_cleanup_on_exit.append(fast_cache_dir)
            print(f"Using {fast_cache_dir} for caching during this run.")
            return fast_cache_path
        else:
            print(f"Using {self.cache_dir} for caching during this run.")
            return self.cache_dir
        
    def cleanup(self):
        try:
            print("Cleaning up cache...")
            for source, dest in self.to_copy_to_cache:
                if os.path.exists(source):
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                    print(f"Copied {source} to {dest}")
            for file in self.to_cleanup_on_exit:
                if os.path.exists(file):
                    shutil.rmtree(file) if os.path.isdir(file) else os.remove(file)  
                    print(f"Removed {file}")
        except Exception as e:
            print(f"Error during cache cleanup: {e}")

    def copy_to_permanent_cache(self):
        for source, dest in self.to_copy_to_cache:
            print(f"Copying {source} to permanent cache location {dest}...")
            if os.path.exists(source):
                shutil.copytree(source, dest, dirs_exist_ok=True)
                print(f"Copied {source} to {dest}")
            else:
                print(f"Warning: Expected file {source} not found during copy to permanent cache.")
