import React from 'react';
import { BG_DARK, BG_MEDIUM } from '../constants/theme';
import { IMAGE_STYLES, ASPECT_RATIOS, getRatioDetails } from '../constants/generator';
import { PlatformImageSettings } from '../hooks/useImageSettings';

interface ImageControlPanelProps {
  mode: 'GENERATE' | 'UPLOAD' | 'NONE';
  style: string;
  aspectRatio: string;
  imagePrompt: string;
  isGeneratingImage: boolean;
  onModeChange: (mode: 'GENERATE' | 'UPLOAD' | 'NONE') => void;
  onStyleChange: (style: string) => void;
  onAspectRatioChange: (ratio: string) => void;
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onGenerateImage: () => void;
  platformKey: string;
}

const ImageControlPanel: React.FC<ImageControlPanelProps> = ({
  mode,
  style,
  aspectRatio,
  imagePrompt,
  isGeneratingImage,
  onModeChange,
  onStyleChange,
  onAspectRatioChange,
  onFileUpload,
  onGenerateImage,
  platformKey,
}) => {
  return (
    <div className="mb-4 p-4 rounded-xl border border-slate-700 space-y-3">
      <p className="text-sm font-semibold text-blue-300">Visual Asset Selection</p>
      
      {/* Mode Selector */}
      <div className="flex justify-between space-x-2 text-sm">
        {(['GENERATE', 'UPLOAD', 'NONE'] as const).map(option => (
          <label 
            key={option} 
            className={`flex-1 p-2 rounded-lg cursor-pointer transition text-center font-medium 
              ${mode === option 
                ? 'bg-blue-600 text-white shadow-md' 
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
          >
            <input 
              type="radio" 
              name={`image-mode-${platformKey}`} 
              value={option} 
              checked={mode === option}
              onChange={() => onModeChange(option)}
              className="hidden"
            />
            {option === 'GENERATE' ? 'Generate New' : option === 'UPLOAD' ? 'Upload Image' : 'No Image'}
          </label>
        ))}
      </div>
      
      {/* Conditional Inputs */}
      {mode === 'GENERATE' && (
        <div className="space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="flex-1 flex flex-col">
              <label className="text-xs text-slate-400 mb-1">Visual Style:</label>
              <select
                value={style}
                onChange={(e) => onStyleChange(e.target.value)}
                className={`p-2 ${BG_DARK} border border-slate-600 rounded-lg text-white text-sm`}
              >
                {IMAGE_STYLES.map(s => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex-1 flex flex-col">
              <label className="text-xs text-slate-400 mb-1">Aspect Ratio:</label>
              <select
                value={aspectRatio}
                onChange={(e) => onAspectRatioChange(e.target.value)}
                className={`p-2 ${BG_DARK} border border-slate-600 rounded-lg text-white text-sm`}
              >
                {ASPECT_RATIOS.map(r => (
                  <option key={r.value} value={r.value}>
                    {r.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <button
            onClick={onGenerateImage}
            disabled={isGeneratingImage || !imagePrompt.trim()}
            className={`w-full px-4 py-2 text-sm font-semibold rounded-lg transition ${
              isGeneratingImage 
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed' 
                : 'bg-emerald-500 text-white hover:bg-emerald-600'
            }`}
          >
            {isGeneratingImage ? 'Generating Visual...' : 'Generate Visual'}
          </button>
        </div>
      )}

      {mode === 'UPLOAD' && (
        <div className="flex flex-col">
          <label className="text-xs text-slate-400 mb-1">Upload File (Max 1MB):</label>
          <input
            type="file"
            accept="image/png, image/jpeg, image/webp"
            onChange={onFileUpload}
            className={`w-full p-2 block ${BG_DARK} border border-slate-600 rounded-lg text-white text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600`}
          />
        </div>
      )}
    </div>
  );
};

export default ImageControlPanel;

