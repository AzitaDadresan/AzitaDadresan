import React, { useState, useEffect } from 'react';
import { BG_DARK, BG_MEDIUM } from '../constants/theme';
import { BG_SLATE_750 } from '../constants/theme';
import { getRatioDetails } from '../constants/generator';
import { PlatformImageSettings, ImageSettings } from '../hooks/useImageSettings';
import SchedulerModal from './SchedulerModal';
import ImageControlPanel from './ImageControlPanel';

interface ContentOutputProps {
  title: string;
  content: string;
  setContent: (content: string) => void;
  colorClass: string;
  platformKey: 'linkedin' | 'wordpress' | 'instagram';
  setGlobalAlert: (message: string) => void;
  imageSettings: PlatformImageSettings;
  handleImageSettingChange: (platform: keyof ImageSettings, field: string, value: any) => void;
  saveScheduledPost: (platform: string, content: string, date: string, timeKey: string, imageUrl: string | null) => void;
  imagePrompt: string;
  isScheduled?: boolean;
}

const platformIcons = {
  linkedin: <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24"><path d="M19 3a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h14zM8 10h2.5v2.5H8V10zm4.5 0h3v2.5h-3V10zM17 10h-2.5v2.5H17V10zM5 10h2.5v2.5H5V10zM8 13.5h2.5v2.5H8v-2.5zm4.5 0h3v2.5h-3v-2.5zM17 13.5h-2.5v2.5H17v-2.5zM5 13.5h2.5v2.5H5v-2.5zm3 3h2.5v2.5H8V16zm4.5 0h3v2.5h-3V16zM17 16h-2.5v2.5H17V16zM5 16h2.5v2.5H5V16zM5 7a2 2 0 00-2 2v2h18V9a2 2 0 00-2-2H5z"/></svg>,
  wordpress: <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8zM9.5 7.5C9.5 6.67 8.83 6 8 6S6.5 6.67 6.5 7.5 7.17 9 8 9s1.5-.67 1.5-1.5zM16 17c-2.45 0-4.52-1.75-4.9-4.05L11.5 12c.38-2.3 2.45-4.05 4.9-4.05 2.53 0 4.58 2.05 4.58 4.58S18.53 17 16 17z"/></svg>,
  instagram: <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-13a5 5 0 100 10 5 5 0 000-10zm0 8a3 3 0 110-6 3 3 0 010 6zm6.5-7.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"/></svg>,
};

const ContentOutput: React.FC<ContentOutputProps> = ({
  title,
  content,
  setContent,
  colorClass,
  platformKey,
  setGlobalAlert,
  imageSettings,
  handleImageSettingChange,
  saveScheduledPost,
  imagePrompt,
  isScheduled = false,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [isRefining, setIsRefining] = useState(false);
  const [showScheduler, setShowScheduler] = useState(false);
  const [refinementInstruction, setRefinementInstruction] = useState('');
  const [currentContent, setCurrentContent] = useState(content);
  const [isContentReadyForSchedule, setIsContentReadyForSchedule] = useState(false);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);

  const { mode, style, imageDataUrl, generatedUrl, aspectRatio } = imageSettings;
  const finalImageUrl = mode === 'GENERATE' ? generatedUrl : (mode === 'UPLOAD' ? imageDataUrl : null);
  const ratioDetails = getRatioDetails(aspectRatio);

  useEffect(() => {
    setCurrentContent(content);
    if (content === "") {
      setIsContentReadyForSchedule(false);
      setShowScheduler(false);
    }
  }, [content]);

  const handleEdit = () => {
    setIsEditing(true);
    setIsRefining(false);
  };

  const handleSave = () => {
    setContent(currentContent);
    setIsEditing(false);
    setIsContentReadyForSchedule(true);
    setGlobalAlert(`${title} content saved successfully. Ready to schedule!`);
  };

  const handleRefineClick = () => {
    setIsRefining(true);
    setIsEditing(false);
  };

  const handleApplyRefinement = () => {
    if (!refinementInstruction.trim()) {
      setGlobalAlert('Please enter a refinement instruction.');
      return;
    }

    const refinement = `\n\n[LLM Refinement applied for ${title} based on: "${refinementInstruction}".]`;
    const refinedText = currentContent.includes('LLM Refinement') 
      ? currentContent.replace(/\[LLM Refinement applied.*\]/s, refinement) 
      : currentContent + refinement;

    setCurrentContent(refinedText);
    setRefinementInstruction('');
    setIsRefining(false);
    setIsEditing(false);
    setIsContentReadyForSchedule(true);
    setGlobalAlert(`${title} content refined and saved! Ready for scheduling.`);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 1024 * 1024) {
        setGlobalAlert('Image size exceeds 1MB limit. Please choose a smaller file.');
        event.target.value = '';
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        handleImageSettingChange(platformKey, 'imageDataUrl', e.target?.result);
        handleImageSettingChange(platformKey, 'generatedUrl', null);
        setGlobalAlert(`${title} image uploaded successfully!`);
      };
      reader.onerror = () => {
        setGlobalAlert('Error reading file.');
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerateImage = async () => {
    if (!imagePrompt.trim()) {
      setGlobalAlert('Please enter a descriptive Image Generation Prompt in the main section.');
      return;
    }

    setIsGeneratingImage(true);
    handleImageSettingChange(platformKey, 'generatedUrl', null);

    await new Promise(resolve => setTimeout(resolve, 2500));

    const mockImageText = `${platformKey} | ${imagePrompt.substring(0, 15)} | ${style.split(' ')[0]} | ${ratioDetails.value}`;
    const generatedUrl = `https://placehold.co/${ratioDetails.size}/${ratioDetails.color}/ffffff?text=${encodeURIComponent(mockImageText)}`;

    handleImageSettingChange(platformKey, 'generatedUrl', generatedUrl);
    setIsGeneratingImage(false);
    setGlobalAlert(`New image visual generated for ${title}!`);
  };

  const handleSchedulePost = (platform: string, content: string, date: string, timeKey: string, imageUrl: string | null) => {
    setShowScheduler(false);
    saveScheduledPost(platform, content, date, timeKey, imageUrl);
  };

  const Icon = platformIcons[platformKey];

  return (
    <div className={`${BG_MEDIUM} p-6 rounded-2xl shadow-xl border-t-4 ${colorClass} transition duration-300 h-full flex flex-col`}>
      {showScheduler && (
        <SchedulerModal 
          title={title}
          onClose={() => setShowScheduler(false)}
          onSchedule={handleSchedulePost}
          currentContent={currentContent}
          generatedImageUrl={finalImageUrl}
        />
      )}

      <h3 className="text-xl font-semibold mb-4 flex items-center text-white">
        {Icon}
        {title} Content
      </h3>
      
      <ImageControlPanel
        mode={mode}
        style={style}
        aspectRatio={aspectRatio}
        imagePrompt={imagePrompt}
        isGeneratingImage={isGeneratingImage}
        onModeChange={(newMode) => {
          handleImageSettingChange(platformKey, 'mode', newMode);
          handleImageSettingChange(platformKey, 'imageDataUrl', null);
          handleImageSettingChange(platformKey, 'generatedUrl', null);
        }}
        onStyleChange={(newStyle) => handleImageSettingChange(platformKey, 'style', newStyle)}
        onAspectRatioChange={(newRatio) => handleImageSettingChange(platformKey, 'aspectRatio', newRatio)}
        onFileUpload={handleFileUpload}
        onGenerateImage={handleGenerateImage}
        platformKey={platformKey}
      />

      {/* Image Preview */}
      {finalImageUrl || isGeneratingImage ? (
        <div className="mb-4 p-3 bg-slate-700 rounded-xl border border-emerald-500/50">
          <p className="text-xs font-medium text-emerald-400 mb-2">
            Image Preview (Ratio: {ratioDetails.value})
          </p>
          {isGeneratingImage ? (
            <div className="w-full h-48 flex items-center justify-center rounded-lg bg-slate-700/70 border border-dashed border-emerald-400 animate-pulse">
              <p className="text-emerald-300 font-semibold text-lg">Loading Visual...</p>
            </div>
          ) : (
            <img 
              src={finalImageUrl || ''} 
              alt="Post Visual" 
              className="w-full h-auto rounded-lg object-cover shadow-lg" 
              style={{ aspectRatio: ratioDetails.value.replace(':', '/') }}
              onError={(e) => {
                if (mode === 'GENERATE') {
                  (e.target as HTMLImageElement).src = `https://placehold.co/${ratioDetails.size}/cc3300/ffffff?text=Image+Load+Error`;
                }
              }}
            />
          )}
        </div>
      ) : (
        <div className="mb-4 p-6 text-center rounded-xl border border-dashed border-slate-700 text-slate-500">
          <p className="text-sm font-medium">No image selected for this post.</p>
        </div>
      )}

      {/* Content Area */}
      {(isEditing || isRefining) ? (
        <textarea
          value={currentContent}
          onChange={(e) => setCurrentContent(e.target.value)}
          className={`flex-grow w-full p-3 mb-4 rounded-lg text-white ${BG_DARK} border border-slate-700 focus:ring-2 focus:ring-blue-500 resize-none font-mono text-sm`}
          rows={10}
        />
      ) : (
        <div className={`flex-grow w-full p-4 mb-4 rounded-lg ${BG_DARK} text-white whitespace-pre-wrap leading-relaxed border border-slate-700`}>
          {currentContent || "Click 'Generate Master Content' above to begin."}
        </div>
      )}

      {/* Refinement Chat Interface */}
      {isRefining && (
        <div className={`p-4 rounded-xl mb-4 space-y-3 border border-emerald-500/50 ${BG_SLATE_750}`}>
          <p className="text-sm font-semibold text-emerald-400">LLM Refinement Chat:</p>
          <textarea
            value={refinementInstruction}
            onChange={(e) => setRefinementInstruction(e.target.value)}
            placeholder="E.g., Make it shorter and add three relevant hashtags."
            className={`w-full p-3 rounded-lg ${BG_DARK} text-white border border-slate-700 resize-none`}
            rows={2}
          />
          <button 
            onClick={handleApplyRefinement}
            className="w-full px-4 py-2 bg-emerald-600 text-white font-semibold rounded-lg hover:bg-emerald-700 transition"
          >
            Apply Refinement
          </button>
        </div>
      )}
      
      {/* Control Buttons */}
      <div className="flex justify-between items-center mt-auto">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {(isEditing || isRefining) ? (
            <button 
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition"
            >
              Save Content
            </button>
          ) : (
            <button 
              onClick={handleEdit}
              disabled={!content || isScheduled}
              className={`px-4 py-2 font-semibold rounded-lg transition ${
                !content || isScheduled 
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              Edit
            </button>
          )}
          
          <button 
            onClick={handleRefineClick}
            disabled={!content || isRefining || isScheduled}
            className={`px-4 py-2 font-semibold rounded-lg transition ${
              !content || isRefining || isScheduled 
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                : 'bg-slate-700 text-blue-300 hover:bg-slate-600'
            }`}
          >
            Refine with LLM
          </button>
        </div>

        {isScheduled ? (
          <div className="px-4 py-2 bg-emerald-700/50 text-emerald-300 font-semibold rounded-lg border border-emerald-600/50">
            ✓ Scheduled
          </div>
        ) : (
          isContentReadyForSchedule && !isEditing && !isRefining && (
            <button
              onClick={() => setShowScheduler(true)}
              className="px-4 py-2 bg-emerald-500 text-white font-bold rounded-lg hover:bg-emerald-600 transition shadow-md shadow-emerald-500/30"
            >
              Schedule Post
            </button>
          )
        )}
      </div>
    </div>
  );
};

export default ContentOutput;

