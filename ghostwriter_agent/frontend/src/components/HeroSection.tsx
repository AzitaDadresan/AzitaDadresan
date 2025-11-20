import React from 'react';
import GeneratorSnapshot from './GeneratorSnapshot';

const HeroSection: React.FC = () => {
  return (
    <section className="bg-gradient-to-br from-slate-900 to-slate-800/50 pt-16 pb-24 px-4 md:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Hero Text */}
        <div className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-6 leading-tight">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
              Generate, Schedule, and Innovate
            </span>
          </h1>
          <p className="text-xl text-slate-300 mb-8 max-w-3xl mx-auto">
            Ghostwriter automates your content creation across LinkedIn, WordPress, and Instagram. 
            Create master content, refine for each platform, and schedule posts—all in one powerful workflow.
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-lg text-slate-400">
            <span className="flex items-center">
              <span className="text-emerald-400 mr-2">✓</span> AI-Powered Content Generation
            </span>
            <span className="flex items-center">
              <span className="text-emerald-400 mr-2">✓</span> Multi-Platform Optimization
            </span>
            <span className="flex items-center">
              <span className="text-emerald-400 mr-2">✓</span> Smart Scheduling
            </span>
          </div>
        </div>

        {/* Generator Snapshot */}
        <GeneratorSnapshot />
      </div>
    </section>
  );
};

export default HeroSection;

