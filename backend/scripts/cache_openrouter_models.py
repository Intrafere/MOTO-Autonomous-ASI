#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporary script to fetch OpenRouter models and cache them for profile matching.
This script will be deleted after execution.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.shared.openrouter_client import OpenRouterClient
from backend.shared.config import rag_config, system_config


async def cache_models():
    """Fetch OpenRouter models and cache them as JSON."""
    
    # Try to get API key from environment first, then from rag_config
    api_key = os.environ.get('OPENROUTER_API_KEY') or rag_config.openrouter_api_key
    if not api_key:
        print("ERROR: No OpenRouter API key configured")
        print("Set OPENROUTER_API_KEY environment variable or configure in rag_config")
        return False
    
    print(f"Fetching OpenRouter models with API key...")
    
    client = OpenRouterClient(api_key)
    try:
        models = await client.list_models(free_only=False)
        
        if not models:
            print("ERROR: No models retrieved from OpenRouter")
            return False
        
        print(f"Retrieved {len(models)} models from OpenRouter")
        
        # Create mapping: display_name -> api_id
        model_mapping = {}
        for model in models:
            model_id = model.get('id', '')
            model_name = model.get('name', '')
            context_length = model.get('context_length', 0)
            
            if model_id and model_name:
                # Create display name with context info
                context_info = f" ({int(context_length/1000)}K)" if context_length else ""
                display_name = f"{model_name}{context_info}"
                
                # Map both the display name and the API ID
                model_mapping[display_name] = model_id
                model_mapping[model_id] = model_id  # Direct mapping for API IDs
                
                print(f"  {display_name} -> {model_id}")
        
        # Cache to JSON
        cache_file = Path(system_config.data_dir) / "model_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(model_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"Cached {len(model_mapping)} model mappings to {cache_file}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to fetch models: {e}")
        return False
    finally:
        await client.close()


async def main():
    """Main entry point."""
    success = await cache_models()
    
    # Delete this script
    script_path = Path(__file__)
    if script_path.exists():
        script_path.unlink()
        print(f"Deleted {script_path}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

