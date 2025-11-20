import { useState } from 'react';
import { StyleProfileFormHandlers, CTASectionHandlers } from '../types';

export const useLandingPage = () => {
  // State for the main Style Profile creation form
  const [projectUrl, setProjectUrl] = useState<string>('');
  const [projectId, setProjectId] = useState<string>('');
  const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
  
  // State for the demo request form
  const [demoEmail, setDemoEmail] = useState<string>('');
  const [isDemoSubmitting, setIsDemoSubmitting] = useState<boolean>(false);

  // State for the global alert message
  const [alertMessage, setAlertMessage] = useState<string>('');

  const handleStyleProfileCreation = () => {
    if (projectUrl.trim() && projectId.trim()) {
      setIsModelLoading(true);
      setAlertMessage('');
      
      // Simulate API call and processing
      setTimeout(() => {
        setIsModelLoading(false);
        setAlertMessage(`Style Profile '${projectId}' successfully initiated using data from: ${projectUrl}. Check the Calibration Dashboard.`);
        setProjectUrl('');
        setProjectId(''); 
      }, 1500);
    } else {
      setAlertMessage('Both the WordPress URL and a Project ID are required to begin calibration.');
    }
  };

  const handleDemoRequest = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (demoEmail.trim()) {
      setIsDemoSubmitting(true);
      setAlertMessage('');
      
      // Simple check for email format
      if (!demoEmail.includes('@') || !demoEmail.includes('.')) {
        setIsDemoSubmitting(false);
        setAlertMessage('Please enter a valid email address.');
        return;
      }

      // Simulate API call for demo request
      setTimeout(() => {
        setIsDemoSubmitting(false);
        setAlertMessage(`Thank you for requesting a demo! We've sent details to ${demoEmail}.`);
        setDemoEmail('');
      }, 1500);
    } else {
      setAlertMessage('Email address is required for the demo request.');
    }
  };

  const handleCloseAlert = () => setAlertMessage('');

  const styleProfileFormHandlers: StyleProfileFormHandlers = {
    projectUrl,
    projectId,
    isModelLoading,
    setProjectUrl,
    setProjectId,
    handleStyleProfileCreation,
  };

  const ctaSectionHandlers: CTASectionHandlers = {
    demoEmail,
    isDemoSubmitting,
    setDemoEmail,
    handleDemoRequest,
  };

  return {
    alertMessage,
    handleCloseAlert,
    styleProfileFormHandlers,
    ctaSectionHandlers,
  };
};

