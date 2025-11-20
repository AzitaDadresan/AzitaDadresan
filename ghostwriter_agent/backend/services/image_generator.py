"""Image generator service using nanobanana API."""
import os
import requests
from typing import Dict, Optional


def generate_image(prompt: str, style: Optional[str] = None) -> Dict:
    """
    Generate an image using nanobanana API.
    
    Args:
        prompt: Text description of the image to generate
        style: Optional style parameter
        
    Returns:
        Dictionary with image_url or error message
    """
    api_key = os.getenv("NANOBANANA_API_KEY")
    
    if not api_key:
        return {
            "success": False,
            "error": "NANOBANANA_API_KEY not configured in environment variables"
        }
    
    try:
        # Nanobanana API endpoint (adjust URL if different)
        api_url = os.getenv("NANOBANANA_API_URL", "https://api.nanobanana.com/v1/generate")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "style": style or "default"
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "image_url": data.get("image_url") or data.get("url"),
                "image_data": data.get("image_data"),  # base64 if provided
                "metadata": data.get("metadata", {})
            }
        else:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

