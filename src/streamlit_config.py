import os
import sys
import platform
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import and patch Streamlit's watcher
from streamlit.watcher import local_sources_watcher
from custom_watcher import CustomLocalSourcesWatcher

# Replace Streamlit's default watcher with our custom one
local_sources_watcher.LocalSourcesWatcher = CustomLocalSourcesWatcher

# Disable Streamlit's file watcher and other potentially problematic features
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Set environment variables for PyTorch
os.environ["PYTORCH_JIT"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if platform.system() == 'Windows':
    # Import asyncio only on Windows
    import asyncio
    import nest_asyncio
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    # Set event loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create and set a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop) 