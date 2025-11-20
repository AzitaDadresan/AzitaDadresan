// Generator-specific constants

// Image Style Options
export const IMAGE_STYLES = [
  { value: 'Professional Portrait', label: 'Professional Portrait: Clean & focused' },
  { value: 'Editorial Graphic', label: 'Editorial Graphic: Abstract & bold' },
  { value: 'Vibrant Social Photo', label: 'Vibrant Social Photo: High contrast' },
  { value: 'Minimalist Iconography', label: 'Minimalist Iconography: Simple flat design' },
];

// Aspect Ratio Options
export const ASPECT_RATIOS = [
  { value: '2:1', label: '2:1 (Link/Banner, LinkedIn)', size: '1200x600', color: '1e3a8a' },
  { value: '16:9', label: '16:9 (Landscape, WordPress)', size: '1200x675', color: '065f46' },
  { value: '1:1', label: '1:1 (Square, Instagram)', size: '1080x1080', color: 'f97316' },
  { value: '4:5', label: '4:5 (Vertical, Instagram/Pins)', size: '1080x1350', color: '9333ea' },
];

export const getRatioDetails = (ratioValue: string) => 
  ASPECT_RATIOS.find(r => r.value === ratioValue) || ASPECT_RATIOS[0];

// Tone Options
export const TONE_OPTIONS = [
  'Informative and Professional',
  'Witty and Engaging',
  'Academic and Data-Driven',
  'Casual and Enthusiastic',
];

// Firebase Config Placeholders
export const getFirebaseConfig = () => {
  const appId = typeof (window as any).__app_id !== 'undefined' ? (window as any).__app_id : 'default-app-id';
  const firebaseConfig = typeof (window as any).__firebase_config !== 'undefined' 
    ? JSON.parse((window as any).__firebase_config) 
    : {};
  const initialAuthToken = typeof (window as any).__initial_auth_token !== 'undefined' 
    ? (window as any).__initial_auth_token 
    : null;
  
  return { appId, firebaseConfig, initialAuthToken };
};

