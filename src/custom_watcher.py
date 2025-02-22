from streamlit.watcher import local_sources_watcher
import sys
import logging

class CustomLocalSourcesWatcher(local_sources_watcher.LocalSourcesWatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_sys_modules = {}
        
    def get_module_paths(self):
        """Override to exclude problematic modules."""
        paths = set()
        
        for name, module in sys.modules.items():
            # Skip PyTorch and related modules
            if name.startswith(('torch', 'sentence_transformers')):
                continue
                
            try:
                if hasattr(module, '__file__') and module.__file__:
                    paths.add(module.__file__)
            except Exception:
                # Skip any modules that cause problems
                continue
                
        return list(paths) 