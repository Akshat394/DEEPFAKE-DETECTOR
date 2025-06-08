import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import VideoUpload from './components/VideoUpload';
import AnalysisDashboard from './components/AnalysisDashboard';
import SystemMonitor from './components/SystemMonitor';
import { AnalysisResult } from './types';

function App() {
  const [currentView, setCurrentView] = useState<'upload' | 'analysis' | 'monitor'>('upload');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisResult(result);
    setCurrentView('analysis');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <Header currentView={currentView} setCurrentView={setCurrentView} />
      
      <main className="container mx-auto px-4 py-8">
        <AnimatePresence mode="wait">
          {currentView === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <VideoUpload onAnalysisComplete={handleAnalysisComplete} />
            </motion.div>
          )}
          
          {currentView === 'analysis' && analysisResult && (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <AnalysisDashboard result={analysisResult} />
            </motion.div>
          )}
          
          {currentView === 'monitor' && (
            <motion.div
              key="monitor"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <SystemMonitor />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;