import React from 'react';
import { BG_DARK } from '../constants/theme';
import { useLandingPage } from '../hooks/useLandingPage';
import CustomAlert from './CustomAlert';
import Header from './Header';
import HeroSection from './HeroSection';
import FeaturesSection from './FeaturesSection';
import OutputChannelsSection from './OutputChannelsSection';
import CTASection from './CTASection';
import Footer from './Footer';

const LandingPage: React.FC = () => {
  const { alertMessage, handleCloseAlert, ctaSectionHandlers } = useLandingPage();

  return (
    <div className={`min-h-screen ${BG_DARK}`}>
      <CustomAlert message={alertMessage} onClose={handleCloseAlert} />
      <Header />
      <HeroSection />
      <FeaturesSection />
      <OutputChannelsSection />
      <CTASection {...ctaSectionHandlers} />
      <Footer />
    </div>
  );
};

export default LandingPage;
