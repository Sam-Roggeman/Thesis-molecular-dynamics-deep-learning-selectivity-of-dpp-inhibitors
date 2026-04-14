import os
import shutil

def construct_file_name(used_percentage, splitname, extension="arrow", prefix="mapped"):
    return f"{prefix}_{used_percentage}_{splitname}.{extension}"
    
class cacheManager:
    to_cleanup_on_exit: list = []
    to_move_to_cache_on_exit: list = []
    def __init__(self, cache_dir, fast_cache_dir=None):
        self.cache_dir = cache_dir
        self.fast_cache_dir = fast_cache_dir

    def get_file_path(self, filename):
        cache_path = os.path.join(self.cache_dir, filename)
        fast_cache_path = os.path.join(self.fast_cache_dir, filename)
        if self.fast_cache_dir is not None:
            # check if file exists in cache dir
            os.makedirs(self.fast_cache_dir, exist_ok=True)
            if os.path.exists(cache_path):
                # copy to fast cache dir
                print(f"Copying {cache_path} to {fast_cache_path} for faster access...")
                shutil.copy(cache_path, fast_cache_path)
            else:
                self.to_move_to_cache_on_exit.append((fast_cache_path, cache_path))
                print(f"File {cache_path} does not exist. Will move {fast_cache_path} to {cache_path} on exit.")
            self.to_cleanup_on_exit.append(fast_cache_path)
            print(f"Using {fast_cache_path} for caching during this run.")
            return fast_cache_path
        else:
            print(f"Using {cache_path} for caching during this run.")
            return cache_path
        
    def cleanup(self):
        try:
            print("Cleaning up cache...")
            for source, dest in self.to_move_to_cache_on_exit:
                if os.path.exists(source):
                    shutil.move(source, dest)
                    print(f"Moved {source} to {dest}")
            for file in self.to_cleanup_on_exit:
                if os.path.exists(file):
                    shutil.rmtree(file) if os.path.isdir(file) else os.remove(file)  
                    print(f"Removed {file}")
        except Exception as e:
            print(f"Error during cache cleanup: {e}")
            