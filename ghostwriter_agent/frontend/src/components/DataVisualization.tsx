import React from 'react';
import { DataIcon } from '../icons';
import { PRIMARY_BLUE, ACCENT_EMERALD } from '../constants/theme';

const DataVisualization: React.FC = () => {
  return (
    <div className="lg:w-5/12 flex justify-center lg:justify-end">
      <div className="w-full max-w-sm md:max-w-md h-64 md:h-80 rounded-3xl p-6 shadow-2xl relative overflow-hidden" 
           style={{ background: 'linear-gradient(135deg, #1e3a8a 0%, #0f172a 70%)', border: '2px solid #3b82f6' }}>
        {/* Simulated Data Flow Lines - CSS now in index.css */}
        <div className="absolute inset-0 opacity-30 ai-visual-react" 
             style={{ backgroundImage: "url('data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100%\\' height=\\'100%\\'><line x1=\\'0\\' y1=\\'0\\' x2=\\'100%\\' y2=\\'100%\\' stroke=\\'rgb(59, 130, 246)\\' stroke-width=\\'1\\'/><line x1=\\'100%\\' y1=\\'0\\' x2=\\'0\\' y2=\\'100%\\' stroke=\\'rgb(16, 185, 129)\\' stroke-width=\\'1\\'/></svg>')" }}></div>
        
        <div className="relative z-10 h-full flex flex-col justify-between text-center">
          <span className={`text-xs font-mono tracking-wider ${PRIMARY_BLUE}`}>
            // Enterprise Style Modeling Protocol
          </span>
          <div className="space-y-2">
            <span className={`text-4xl font-extrabold block ${ACCENT_EMERALD}`}>
              <DataIcon className="w-10 h-10 inline-block mr-2"/>
              Compliance Core Active
            </span>
            <p className="text-sm text-slate-300">Analyzing 1.2M style rules, 99.8% semantic match achieved.</p>
          </div>
          <span className={`text-xs font-mono tracking-wider ${PRIMARY_BLUE}`}>
            // Ready for Scaled Distribution
          </span>
        </div>
      </div>
    </div>
  );
};

export default DataVisualization;

