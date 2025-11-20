import React from 'react';
import { BG_DARK, BG_MEDIUM } from '../constants/theme';
import { TONE_OPTIONS } from '../constants/generator';

interface MasterContentGeneratorProps {
  topic: string;
  tone: string;
  imagePrompt: string;
  isImageGenerationActive: boolean;
  isGenerating: boolean;
  onTopicChange: (topic: string) => void;
  onToneChange: (tone: string) => void;
  onImagePromptChange: (prompt: string) => void;
  onGenerate: () => void;
}

const MasterContentGenerator: React.FC<MasterContentGeneratorProps> = ({
  topic,
  tone,
  imagePrompt,
  isImageGenerationActive,
  isGenerating,
  onTopicChange,
  onToneChange,
  onImagePromptChange,
  onGenerate,
}) => {
  return (
    <section className="bg-gradient-to-br from-slate-900 to-slate-800/50 pt-10 pb-16 px-4 md:px-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-8 leading-tight text-center">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            Precision Content Automation via Style Profile Calibration.
          </span>
        </h1>

        <div className="max-w-4xl mx-auto p-8 rounded-3xl bg-slate-850 shadow-2xl border border-blue-800/50">
          <h2 className="text-2xl font-bold mb-4 text-white">1. Master Content Generation</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <input 
              type="text" 
              value={topic}
              onChange={(e) => onTopicChange(e.target.value)}
              placeholder="E.g., The rise of AI in marketing" 
              className={`col-span-1 md:col-span-2 p-4 ${BG_MEDIUM} border border-blue-600 rounded-xl text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-inner`}
              required
            />
            <select 
              value={tone}
              onChange={(e) => onToneChange(e.target.value)}
              className={`p-4 ${BG_MEDIUM} border border-blue-600 rounded-xl text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-inner`}
            >
              {TONE_OPTIONS.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
          
          <div className={`mb-6 p-4 rounded-xl ${isImageGenerationActive ? 'bg-slate-700 border border-emerald-500/50' : 'bg-slate-800'}`}>
            <label htmlFor="imagePrompt" className={`block text-sm font-semibold mb-2 ${isImageGenerationActive ? 'text-emerald-400' : 'text-slate-400'}`}>
              Image Generation Prompt (Used for all platforms set to 'Generate New')
            </label>
            <input
              type="text"
              id="imagePrompt"
              value={imagePrompt}
              onChange={(e) => onImagePromptChange(e.target.value)}
              placeholder="A detailed description of the image to generate, e.g., 'Minimalist abstract graphic of a glowing blue brain.'"
              className={`w-full p-3 ${BG_DARK} border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-blue-500 shadow-inner`}
            />
          </div>
          
          <button 
            onClick={onGenerate}
            disabled={isGenerating || !topic.trim()}
            className={`w-full px-8 py-4 text-xl font-bold rounded-xl whitespace-nowrap transition-all duration-300 
              ${isGenerating 
                ? 'bg-slate-600 text-slate-400 cursor-not-allowed' 
                : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-lg shadow-emerald-500/40'
              }`}
          >
            {isGenerating ? 'Generating Drafts...' : 'Generate Master Content'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default MasterContentGenerator;

