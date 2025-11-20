import { useState } from 'react';
import { IMAGE_STYLES, ASPECT_RATIOS } from '../constants/generator';

export interface ImageSettings {
  prompt: string;
  linkedin: PlatformImageSettings;
  wordpress: PlatformImageSettings;
  instagram: PlatformImageSettings;
}

export interface PlatformImageSettings {
  mode: 'GENERATE' | 'UPLOAD' | 'NONE';
  style: string;
  imageDataUrl: string | null;
  generatedUrl: string | null;
  aspectRatio: string;
}

const initialImageSettings: ImageSettings = {
  prompt: '',
  linkedin: { mode: 'NONE', style: IMAGE_STYLES[0].value, imageDataUrl: null, generatedUrl: null, aspectRatio: '2:1' },
  wordpress: { mode: 'NONE', style: IMAGE_STYLES[1].value, imageDataUrl: null, generatedUrl: null, aspectRatio: '16:9' },
  instagram: { mode: 'NONE', style: IMAGE_STYLES[2].value, imageDataUrl: null, generatedUrl: null, aspectRatio: '1:1' },
};

export const useImageSettings = () => {
  const [imageSettings, setImageSettings] = useState<ImageSettings>(initialImageSettings);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const handleImageSettingChange = (platform: keyof ImageSettings, field: string, value: any) => {
    if (platform === 'prompt') {
      setImageSettings(prev => ({ ...prev, prompt: value }));
    } else {
      setImageSettings(prev => ({
        ...prev,
        [platform]: {
          ...prev[platform],
          [field]: value
        }
      }));
    }
  };

  const generateImage = async (platform: 'linkedin' | 'wordpress' | 'instagram') => {
    const settings = imageSettings[platform];
    if (settings.mode !== 'GENERATE' || !imageSettings.prompt) {
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch('/api/generate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: imageSettings.prompt,
          style: settings.style,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate image');
      }

      const data = await response.json();
      
      // Update the platform's generatedUrl with the result
      setImageSettings(prev => ({
        ...prev,
        [platform]: {
          ...prev[platform],
          generatedUrl: data.url || data.image_url || null,
        }
      }));

      setIsGenerating(false);
      return data;
    } catch (error) {
      setIsGenerating(false);
      throw error;
    }
  };

  const isImageGenerationActive = ['linkedin', 'wordpress', 'instagram'].some(
    p => imageSettings[p as keyof ImageSettings]?.mode === 'GENERATE'
  );

  return {
    imageSettings,
    handleImageSettingChange,
    generateImage,
    isGenerating,
    isImageGenerationActive,
  };
};

