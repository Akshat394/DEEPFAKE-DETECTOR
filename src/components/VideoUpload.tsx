import React, { useState, useCallback } from 'react';
import { Upload, Video, FileText, Zap, Brain } from 'lucide-react';
import { motion } from 'framer-motion';
import { AnalysisResult } from '../types';

interface VideoUploadProps {
  onAnalysisComplete: (result: AnalysisResult) => void;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onAnalysisComplete }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const processingStages = [
    { stage: 'Frame Extraction', description: 'Extracting frames at 30fps using OpenCV' },
    { stage: 'Face Detection', description: 'MTCNN face detection and alignment' },
    { stage: 'ASCII Conversion', description: 'Novel ASCII pattern transformation (65% compute reduction)' },
    { stage: 'Feature Extraction', description: 'Dual-path InceptionResNetV2 + EfficientNet-B4' },
    { stage: 'Beadal Analysis', description: 'Texture descriptor for compression artifacts' },
    { stage: 'Temporal Fusion', description: 'BiLSTM sequence modeling and feature fusion' },
    { stage: 'Classification', description: 'Sigmoid classifier with confidence scoring' },
  ];

  const runRealAnalysis = async (file: File) => {
    setIsProcessing(true);
    setError(null);

    for (let i = 0; i < processingStages.length; i++) {
      setProcessingStage(processingStages[i].stage);
      await new Promise(resolve => setTimeout(resolve, 350));
    }

    try {
      const formData = new FormData();
      formData.append('file', file);
      const apiBase = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');
      const response = await fetch(`${apiBase}/api/detect`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Detection request failed');
      }
      const result = (await response.json()) as AnalysisResult;
      onAnalysisComplete(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to analyze video';
      setError(message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const videoFile = files.find(file => file.type.startsWith('video/'));
    
    if (videoFile) {
      setSelectedFile(videoFile);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Architecture Overview */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-2xl font-bold text-white mb-4">Dual-Path Detection Architecture</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Video className="h-5 w-5 text-blue-400" />
              <span className="font-medium text-white">Path A: Original</span>
            </div>
            <p className="text-sm text-gray-300">InceptionResNetV2 feature extraction from normalized frames</p>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <FileText className="h-5 w-5 text-green-400" />
              <span className="font-medium text-white">Path B: ASCII</span>
            </div>
            <p className="text-sm text-gray-300">Novel ASCII conversion with EfficientNet-B4 processing</p>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="h-5 w-5 text-purple-400" />
              <span className="font-medium text-white">Temporal Fusion</span>
            </div>
            <p className="text-sm text-gray-300">BiLSTM sequence modeling with attention gates</p>
          </div>
        </div>
      </div>

      {/* File Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
          isDragOver 
            ? 'border-cyan-400 bg-cyan-400/10' 
            : 'border-gray-600 hover:border-gray-500'
        } ${selectedFile ? 'bg-gray-800' : 'bg-gray-800/50'}`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
      >
        {!selectedFile ? (
          <>
            <Upload className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Upload Video for Analysis</h3>
            <p className="text-gray-400 mb-6">
              Supports MP4, AVI, MOV formats. Maximum file size: 500MB
            </p>
            <label className="inline-flex items-center space-x-2 bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-3 rounded-lg cursor-pointer transition-colors">
              <Upload className="h-4 w-4" />
              <span>Select Video File</span>
              <input
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
          </>
        ) : (
          <div className="space-y-4">
            <Video className="h-12 w-12 text-cyan-400 mx-auto" />
            <div>
              <h3 className="text-lg font-semibold text-white">{selectedFile.name}</h3>
              <p className="text-gray-400">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => runRealAnalysis(selectedFile)}
                disabled={isProcessing}
                className="flex items-center space-x-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors"
              >
                <Zap className="h-4 w-4" />
                <span>{isProcessing ? 'Processing...' : 'Start Analysis'}</span>
              </button>
              <button
                onClick={() => setSelectedFile(null)}
                disabled={isProcessing}
                className="px-6 py-3 text-gray-300 hover:text-white transition-colors"
              >
                Remove
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Processing Status */}
      {isProcessing && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 rounded-xl p-6 border border-gray-700"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Processing Video...</h3>
          <div className="space-y-3">
            {processingStages.map((stage, index) => {
              const isActive = stage.stage === processingStage;
              const isComplete = processingStages.findIndex(s => s.stage === processingStage) > index;
              
              return (
                <div
                  key={stage.stage}
                  className={`flex items-center space-x-3 p-3 rounded-lg transition-all ${
                    isActive 
                      ? 'bg-cyan-500/20 border border-cyan-500/30' 
                      : isComplete 
                        ? 'bg-green-500/20 border border-green-500/30' 
                        : 'bg-gray-700/50'
                  }`}
                >
                  <div className={`w-3 h-3 rounded-full ${
                    isActive 
                      ? 'bg-cyan-400 animate-pulse' 
                      : isComplete 
                        ? 'bg-green-400' 
                        : 'bg-gray-500'
                  }`} />
                  <div className="flex-1">
                    <div className="font-medium text-white">{stage.stage}</div>
                    <div className="text-sm text-gray-400">{stage.description}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {error && (
        <div className="bg-red-500/15 border border-red-500/40 text-red-200 rounded-lg p-4 text-sm">
          {error}
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
