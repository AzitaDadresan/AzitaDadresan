import React from 'react';
import { CTASectionHandlers } from '../types';
import { BG_MEDIUM } from '../constants/theme';

const CTASection: React.FC<CTASectionHandlers> = ({
  demoEmail,
  isDemoSubmitting,
  setDemoEmail,
  handleDemoRequest,
}) => {
  return (
    <section className="py-24 px-4 md:px-8 bg-blue-900/40">
      <div className="max-w-3xl mx-auto text-center">
        <h2 className="text-4xl font-extrabold mb-4 text-white">
          Need Dedicated Infrastructure?
        </h2>
        <p className="text-xl text-blue-200 mb-8">
          Request an enterprise demo to integrate Ghostwriter into your dedicated infrastructure or private cloud.
        </p>
        
        {/* Email Submission Form */}
        <form onSubmit={handleDemoRequest} className="flex flex-col sm:flex-row gap-3 max-w-lg mx-auto">
          <input 
            type="email" 
            value={demoEmail}
            onChange={(e) => setDemoEmail(e.target.value)}
            placeholder="Your Corporate Email Address" 
            className={`flex-grow p-4 ${BG_MEDIUM} border border-emerald-600 rounded-xl text-white placeholder-slate-400 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 shadow-lg`}
            required
          />
          <button 
            type="submit"
            disabled={isDemoSubmitting}
            className={`px-10 py-4 text-xl font-bold rounded-xl transition-all duration-300 
              ${isDemoSubmitting 
                ? 'bg-slate-600 cursor-not-allowed' 
                : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-lg shadow-emerald-500/50'
              }`}
          >
            {isDemoSubmitting ? 'Submitting...' : 'Request Demo'}
          </button>
        </form>
      </div>
    </section>
  );
};

export default CTASection;

