import React from 'react';
import { CustomAlertProps } from '../types';

const CustomAlert: React.FC<CustomAlertProps> = ({ message, onClose }) => {
  if (!message) return null;

  return (
    <div 
      className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 p-4 rounded-xl shadow-2xl transition-all duration-300 ease-in-out bg-yellow-600 text-white flex items-center"
      role="alert"
    >
      <span>{message}</span>
      <button 
        onClick={onClose} 
        className="ml-4 font-bold text-xl leading-none opacity-70 hover:opacity-100 transition-opacity"
      >
        &times;
      </button>
    </div>
  );
};

export default CustomAlert;

