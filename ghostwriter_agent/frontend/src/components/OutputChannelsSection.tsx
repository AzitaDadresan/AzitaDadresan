import React from 'react';
import FeatureCard from './FeatureCard';

const OutputChannelsSection: React.FC = () => {
  return (
    <section className="py-20 px-4 md:px-8 bg-slate-900">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-16 text-white">
          Secure, Multi-Channel Deployment
        </h2>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Output 1: Long-Form Articles (WordPress) */}
          <FeatureCard 
            title="Long-Form Articles & Whitepapers" 
            description="Full-length, SEO-driven content delivered directly to your WordPress or custom CMS draft folders. Includes metadata and optimized headings."
            color="blue"
            icon={
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v10a2 2 0 002 2zm0 0l2.167 1.084V18m0 0v-10l-2.167 1.084m2.167 1.084L19 18m-6-10h-2m5-3H7"></path>
              </svg>
            }
          />

          {/* Output 2: Social Micro-Content (LinkedIn/Twitter) */}
          <FeatureCard 
            title="Social Micro-Content Campaigns" 
            description="Generate highly targeted, compliant social posts for professional networks like LinkedIn and X (Twitter). Maintain consistency across all campaigns."
            color="emerald"
            icon={
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M7 15h10m1.5-4.5a3 3 0 11-6 0 3 3 0 016 0zm-12 0a3 3 0 11-6 0 3 3 0 016 0zm10 0a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
            }
          />
        </div>
      </div>
    </section>
  );
};

export default OutputChannelsSection;

