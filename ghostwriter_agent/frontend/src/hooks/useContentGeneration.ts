import { useState } from 'react';

export interface ContentOutputs {
  master: string;
  linkedin: string;
  wordpress: string;
  instagram: string;
}

export const useContentGeneration = () => {
  const [topic, setTopic] = useState<string>('');
  const [tone, setTone] = useState<string>('Informative and Professional');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [contentOutputs, setContentOutputs] = useState<ContentOutputs>({
    master: '',
    linkedin: '',
    wordpress: '',
    instagram: '',
  });

  const generateContent = async (): Promise<string> => {
    if (!topic.trim()) {
      throw new Error('Please enter a topic to begin content generation.');
    }

    setIsGenerating(true);

    try {
      // Call backend content generation endpoint
      const response = await fetch('/api/generate-content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          topic: topic,
          tone: tone
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate content from backend');
      }

      const data = await response.json();
      
      // Use structured outputs from backend
      if (data.outputs) {
        setContentOutputs(data.outputs);
      } else {
        // Fallback if backend doesn't return structured data
        const agentResult = data.result || '';
        const outputs: ContentOutputs = {
          master: `## ${topic}\n\n${agentResult}\n\n**Generated with tone: ${tone}**`,
          linkedin: `💡 ${topic}\n\n${agentResult.substring(0, 200)}...\n\n#${topic.replace(/\s/g, '')} #ContentStrategy #AI`,
          wordpress: `<h1>${topic}</h1>\n\n<p>${agentResult}</p>`,
          instagram: `🔥 ${topic}!\n\n${agentResult.substring(0, 150)}...\n\n#${topic.split(' ')[0]} #AIContent`,
        };
        setContentOutputs(outputs);
      }

      setIsGenerating(false);

      return `Content generated successfully for topic: "${topic}"! Now generate your visuals.`;
    } catch (error) {
      setIsGenerating(false);
      throw error;
    }
  };

  const clearContent = () => {
    setContentOutputs({
      master: '',
      linkedin: '',
      wordpress: '',
      instagram: '',
    });
  };

  return {
    topic,
    setTopic,
    tone,
    setTone,
    isGenerating,
    contentOutputs,
    generateContent,
    clearContent,
  };
};

