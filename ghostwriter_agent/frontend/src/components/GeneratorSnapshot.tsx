import React from 'react';
import { BG_DARK, BG_MEDIUM } from '../constants/theme';
import { PRIMARY_BLUE_CLASS, ACCENT_EMERALD_CLASS } from '../constants/theme';

const GeneratorSnapshot: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto">
      <div className={`${BG_MEDIUM} rounded-2xl p-6 shadow-2xl border border-blue-600/30 overflow-hidden`}>
        {/* Mock Header */}
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-700">
          <div className="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
            Ghostwriter
          </div>
          <div className="flex gap-2">
            <div className="px-3 py-1 bg-blue-600 text-white text-sm font-semibold rounded">Generator</div>
            <div className="px-3 py-1 bg-slate-700 text-slate-300 text-sm font-semibold rounded">Scheduled Posts (3)</div>
          </div>
        </div>

        {/* Mock Content Generation Section */}
        <div className={`${BG_DARK} rounded-xl p-6 mb-6 border border-blue-600/50`}>
          <h3 className="text-xl font-bold mb-4 text-white">1. Master Content Generation</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className={`col-span-2 p-4 ${BG_MEDIUM} border border-blue-600/50 rounded-lg text-slate-400`}>
              The rise of AI in marketing...
            </div>
            <div className={`p-4 ${BG_MEDIUM} border border-blue-600/50 rounded-lg text-slate-400`}>
              Informative and Professional
            </div>
          </div>
          <div className={`w-full px-6 py-4 bg-emerald-500 text-white font-bold rounded-xl text-center`}>
            Generate Master Content
          </div>
        </div>

        {/* Mock Platform Cards */}
        <div className="grid md:grid-cols-3 gap-4">
          {/* LinkedIn Card */}
          <div className={`${BG_DARK} rounded-xl p-4 border-t-4 ${PRIMARY_BLUE_CLASS} shadow-lg`}>
            <div className="flex items-center mb-3">
              <div className="w-6 h-6 bg-blue-500 rounded mr-2"></div>
              <h4 className="text-lg font-semibold text-white">LinkedIn</h4>
            </div>
            <div className={`${BG_MEDIUM} rounded-lg p-3 mb-3 text-sm text-slate-300 min-h-[100px]`}>
              💡 Key Takeaway: The digital revolution makes AI in marketing more vital than ever...
            </div>
            <div className="flex gap-2">
              <div className="px-3 py-1 bg-blue-600 text-white text-xs rounded">Edit</div>
              <div className="px-3 py-1 bg-slate-700 text-blue-300 text-xs rounded">Refine</div>
              <div className="px-3 py-1 bg-emerald-500 text-white text-xs rounded">Schedule</div>
            </div>
          </div>

          {/* WordPress Card */}
          <div className={`${BG_DARK} rounded-xl p-4 border-t-4 ${ACCENT_EMERALD_CLASS} shadow-lg`}>
            <div className="flex items-center mb-3">
              <div className="w-6 h-6 bg-emerald-500 rounded mr-2"></div>
              <h4 className="text-lg font-semibold text-white">WordPress</h4>
            </div>
            <div className={`${BG_MEDIUM} rounded-lg p-3 mb-3 text-sm text-slate-300 min-h-[100px]`}>
              &lt;h1&gt;The Definitive Guide to AI in Marketing Mastery&lt;/h1&gt;...
            </div>
            <div className="flex gap-2">
              <div className="px-3 py-1 bg-blue-600 text-white text-xs rounded">Edit</div>
              <div className="px-3 py-1 bg-slate-700 text-blue-300 text-xs rounded">Refine</div>
              <div className="px-3 py-1 bg-emerald-500 text-white text-xs rounded">Schedule</div>
            </div>
          </div>

          {/* Instagram Card */}
          <div className={`${BG_DARK} rounded-xl p-4 border-t-4 ${PRIMARY_BLUE_CLASS} shadow-lg`}>
            <div className="flex items-center mb-3">
              <div className="w-6 h-6 bg-blue-500 rounded mr-2"></div>
              <h4 className="text-lg font-semibold text-white">Instagram</h4>
            </div>
            <div className={`${BG_MEDIUM} rounded-lg p-3 mb-3 text-sm text-slate-300 min-h-[100px]`}>
              🔥 Trending Topic: AI in marketing! The digital landscape demands...
            </div>
            <div className="flex gap-2">
              <div className="px-3 py-1 bg-blue-600 text-white text-xs rounded">Edit</div>
              <div className="px-3 py-1 bg-slate-700 text-blue-300 text-xs rounded">Refine</div>
              <div className="px-3 py-1 bg-emerald-500 text-white text-xs rounded">Schedule</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GeneratorSnapshot;

