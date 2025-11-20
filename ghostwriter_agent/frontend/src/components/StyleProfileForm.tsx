import React from 'react';
import { StyleProfileFormHandlers } from '../types';
import { BG_MEDIUM } from '../constants/theme';
import WordPressChecker from './WordPressChecker';

const StyleProfileForm: React.FC<StyleProfileFormHandlers> = ({
  projectUrl,
  projectId,
  isModelLoading,
  setProjectUrl,
  setProjectId,
  handleStyleProfileCreation,
}) => {
  return (
    <div className="w-full max-w-xl lg:max-w-full mx-auto lg:mx-0 space-y-4">
      <p className="text-md text-emerald-400 font-semibold text-left">
        Step 1: Calibrate Your Content Style
      </p>
      <div className="flex flex-col gap-3">
        <div>
          <input 
            type="url" 
            value={projectUrl}
            onChange={(e) => setProjectUrl(e.target.value)}
            placeholder="Paste your WordPress URL for source ingestion" 
            className={`w-full p-4 ${BG_MEDIUM} border border-blue-600 rounded-xl text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-lg`}
            required
          />
          <WordPressChecker url={projectUrl} />
        </div>
        <input 
          type="text" 
          value={projectId}
          onChange={(e) => setProjectId(e.target.value)}
          placeholder="Enter a unique Project ID (e.g., 'Q3-Marketing-Profile')" 
          className={`p-4 ${BG_MEDIUM} border border-blue-600 rounded-xl text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-lg`}
          required
        />
        <button 
          onClick={handleStyleProfileCreation}
          disabled={projectUrl.trim() === '' || projectId.trim() === '' || isModelLoading}
          className={`w-full px-8 py-4 font-bold rounded-xl whitespace-nowrap transition-all duration-300 
            ${isModelLoading 
              ? 'bg-slate-600 cursor-not-allowed' 
              : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/40'
            }`}
        >
          {isModelLoading ? 'Analyzing Data...' : 'Generate Style Profile'}
        </button>
      </div>
      <p className="text-sm text-slate-400 text-left">
        <span className="font-semibold text-emerald-400">Calibration in under 60 seconds.</span> Begin your initial setup today.
      </p>
    </div>
  );
};

export default StyleProfileForm;

