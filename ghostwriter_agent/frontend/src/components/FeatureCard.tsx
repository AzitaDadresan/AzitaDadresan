import React from 'react';
import { FeatureCardProps } from '../types';
import { BG_MEDIUM, TEXT_MUTED } from '../constants/theme';

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description, color }) => (
  <div className={`${BG_MEDIUM} p-8 rounded-2xl shadow-xl transition hover:shadow-xl hover:shadow-${color}-800/30 border-t-4 border-${color}-600/70`}>
    <div className="flex items-start space-x-4">
      <div className={`w-10 h-10 ${color} flex-shrink-0 mt-1`}>
        {icon}
      </div>
      <div>
        <h3 className="text-2xl font-semibold mb-2 text-white">{title}</h3>
        <p className={TEXT_MUTED}>
          {description}
        </p>
      </div>
    </div>
  </div>
);

export default FeatureCard;

