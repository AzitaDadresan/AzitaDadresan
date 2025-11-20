import React from 'react';
import { BG_MEDIUM, PRIMARY_BLUE, ACCENT_EMERALD, TEXT_MUTED } from '../constants/theme';

const FeaturesSection: React.FC = () => {
  return (
    <section className="py-20 px-4 md:px-8">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-12 text-white">
          Pillars of Enterprise Content
        </h2>

        <div className="grid md:grid-cols-3 gap-10">
          {/* Feature 1: Style Compliance */}
          <div className={`${BG_MEDIUM} p-8 rounded-2xl shadow-xl border-t-4 border-blue-600/70`}>
            <span className={`text-5xl font-extrabold block mb-4 ${PRIMARY_BLUE}`}>
              <svg className="w-8 h-8 inline-block" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.515a12.031 12.031 0 011.533 1.148 10.05 10.05 0 01-5.139 12.235 10.05 10.05 0 01-12.235-5.139 12.031 12.031 0 011.148 1.533 10.05 10.05 0 0112.235 5.139 10.05 10.05 0 01-5.139-12.235z"></path>
              </svg>
            </span>
            <h3 className="text-xl font-semibold mb-3 text-white">Guaranteed Style Compliance</h3>
            <p className={TEXT_MUTED}>
              Train our deep learning model on your entire existing content library. We guarantee output that adheres to your specific brand tone, vocabulary, and internal style guides.
            </p>
          </div>

          {/* Feature 2: Editorial Accuracy */}
          <div className={`${BG_MEDIUM} p-8 rounded-2xl shadow-xl border-t-4 border-emerald-600/70`}>
            <span className={`text-5xl font-extrabold block mb-4 ${ACCENT_EMERALD}`}>
              <svg className="w-8 h-8 inline-block" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path>
              </svg>
            </span>
            <h3 className="text-xl font-semibold mb-3 text-white">Fact-Checked & Grounded</h3>
            <p className={TEXT_MUTED}>
              Every piece of content is grounded against internal data sources and the latest public information, minimizing risk and ensuring factual accuracy before publication.
            </p>
          </div>

          {/* Feature 3: Scalability */}
          <div className={`${BG_MEDIUM} p-8 rounded-2xl shadow-xl border-t-4 border-blue-600/70`}>
            <span className={`text-5xl font-extrabold block mb-4 ${PRIMARY_BLUE}`}>
              <svg className="w-8 h-8 inline-block" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
              </svg>
            </span>
            <h3 className="text-xl font-semibold mb-3 text-white">Seamless Workflow Integration</h3>
            <p className={TEXT_MUTED}>
              Scale your output by 10x without sacrificing quality. Ghostwriter integrates directly with your CMS, marketing, and internal review systems for rapid, audited deployment.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;

