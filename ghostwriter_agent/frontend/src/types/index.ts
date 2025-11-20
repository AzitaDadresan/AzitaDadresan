// Type Definitions
export interface CustomAlertProps {
  message: string;
  onClose: () => void;
}

export interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: string;
}

export interface DataIconProps {
  className?: string;
}

export interface WordPressCheckResult {
  is_wordpress: boolean;
  score: number;
  signals: {
    wp_json: boolean;
    wp_content: boolean;
    wp_login: boolean;
    meta_generator: boolean;
    headers_powered: boolean;
  };
}

export interface StyleProfileFormHandlers {
  projectUrl: string;
  projectId: string;
  isModelLoading: boolean;
  setProjectUrl: (url: string) => void;
  setProjectId: (id: string) => void;
  handleStyleProfileCreation: () => void;
}

export interface CTASectionHandlers {
  demoEmail: string;
  isDemoSubmitting: boolean;
  setDemoEmail: (email: string) => void;
  handleDemoRequest: (e: React.FormEvent<HTMLFormElement>) => void;
}

