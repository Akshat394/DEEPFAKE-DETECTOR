import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Clock, Cpu, Eye, FileText } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { AnalysisResult } from '../types';

interface AnalysisDashboardProps {
  result: AnalysisResult;
}

const AnalysisDashboard: React.FC<AnalysisDashboardProps> = ({ result }) => {
  const [selectedFrame, setSelectedFrame] = useState(0);

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'REAL': return 'text-green-400 bg-green-400/20 border-green-400/30';
      case 'FAKE': return 'text-red-400 bg-red-400/20 border-red-400/30';
      case 'SUSPICIOUS': return 'text-yellow-400 bg-yellow-400/20 border-yellow-400/30';
      default: return 'text-gray-400 bg-gray-400/20 border-gray-400/30';
    }
  };

  const frameData = result.frameAnalysis.map(frame => ({
    frame: frame.frameNumber,
    confidence: frame.confidence * 100,
    pathA: frame.pathAScore * 100,
    pathB: frame.pathBScore * 100,
    fusion: frame.fusionScore * 100,
  }));

  const modelMetricsData = [
    { name: 'InceptionResNetV2', score: result.modelMetrics.inceptionScore * 100 },
    { name: 'EfficientNet-B4', score: result.modelMetrics.efficientNetScore * 100 },
    { name: 'BiLSTM Fusion', score: result.modelMetrics.lstmScore * 100 },
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-4 sm:space-y-6 px-4 sm:px-6">
      {/* Overall Results Header */}
      <div className="bg-gray-800 rounded-xl p-4 sm:p-6 border border-gray-700">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 space-y-3 sm:space-y-0">
          <h2 className="text-xl sm:text-2xl font-bold text-white">Analysis Results</h2>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-sm text-gray-400">Processing Time</div>
              <div className="flex items-center space-x-1">
                <Clock className="h-4 w-4 text-cyan-400" />
                <span className="text-white font-medium">{result.processingTime}s</span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6">
          <div className="text-center">
            <div className="text-2xl sm:text-3xl font-bold text-white mb-1 sm:mb-2">
              {(result.overallScore * 100).toFixed(1)}%
            </div>
            <div className="text-xs sm:text-sm text-gray-400">Overall Score</div>
          </div>
          
          <div className="text-center">
            <div className={`inline-flex items-center px-2 sm:px-3 py-1 rounded-full border text-xs sm:text-sm font-medium ${
              getClassificationColor(result.classification)
            }`}>
              {result.classification === 'REAL' ? (
                <CheckCircle className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />
              ) : (
                <AlertTriangle className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />
              )}
              {result.classification}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl sm:text-3xl font-bold text-white mb-1 sm:mb-2">
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-xs sm:text-sm text-gray-400">Confidence</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl sm:text-3xl font-bold text-cyan-400 mb-1 sm:mb-2">
              {result.modelMetrics.computeReduction.toFixed(1)}%
            </div>
            <div className="text-xs sm:text-sm text-gray-400">Compute Reduction</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Frame-by-Frame Analysis */}
        <div className="bg-gray-800 rounded-xl p-4 sm:p-6 border border-gray-700">
          <h3 className="text-lg sm:text-xl font-semibold text-white mb-4">Frame-by-Frame Confidence</h3>
          <div className="h-48 sm:h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frameData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="frame" 
                  stroke="#9CA3AF"
                  fontSize={10}
                  tick={{ fill: '#9CA3AF' }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  fontSize={10}
                  domain={[0, 100]}
                  tick={{ fill: '#9CA3AF' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#fff',
                    fontSize: '12px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="#06B6D4" 
                  strokeWidth={2}
                  dot={{ fill: '#06B6D4', strokeWidth: 0, r: 2 }}
                  name="Overall Confidence"
                />
                <Line 
                  type="monotone" 
                  dataKey="pathA" 
                  stroke="#3B82F6" 
                  strokeWidth={1}
                  dot={false}
                  name="Path A (Original)"
                />
                <Line 
                  type="monotone" 
                  dataKey="pathB" 
                  stroke="#10B981" 
                  strokeWidth={1}
                  dot={false}
                  name="Path B (ASCII)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Performance */}
        <div className="bg-gray-800 rounded-xl p-4 sm:p-6 border border-gray-700">
          <h3 className="text-lg sm:text-xl font-semibold text-white mb-4">Model Performance</h3>
          <div className="h-48 sm:h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelMetricsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="name" 
                  stroke="#9CA3AF"
                  fontSize={10}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  tick={{ fill: '#9CA3AF' }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  fontSize={10}
                  domain={[0, 100]}
                  tick={{ fill: '#9CA3AF' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#fff',
                    fontSize: '12px'
                  }}
                />
                <Bar 
                  dataKey="score" 
                  fill="#8B5CF6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ASCII Preview & Temporal Anomalies */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* ASCII Preview */}
        <div className="bg-gray-800 rounded-xl p-4 sm:p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <FileText className="h-4 w-4 sm:h-5 sm:w-5 text-green-400" />
            <h3 className="text-lg sm:text-xl font-semibold text-white">ASCII Pattern Preview</h3>
          </div>
          <div className="bg-black rounded-lg p-3 sm:p-4 font-mono text-[10px] sm:text-xs overflow-x-auto">
            {result.asciiPreview.slice(0, 3).map((line, idx) => (
              <div key={idx} className="text-green-400 leading-tight whitespace-pre">
                {line}
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs sm:text-sm text-gray-400">
            Novel ASCII conversion reduces compute by {result.modelMetrics.computeReduction}%
          </div>
        </div>

        {/* Temporal Anomalies */}
        <div className="bg-gray-800 rounded-xl p-4 sm:p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Eye className="h-4 w-4 sm:h-5 sm:w-5 text-yellow-400" />
            <h3 className="text-lg sm:text-xl font-semibold text-white">Temporal Anomalies</h3>
          </div>
          <div className="space-y-2 sm:space-y-3">
            {result.temporalAnomalies.map((anomaly, idx) => (
              <div key={idx} className="bg-gray-700 rounded-lg p-2 sm:p-3">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-2 space-y-1 sm:space-y-0">
                  <span className="text-xs sm:text-sm font-medium text-white capitalize">
                    {anomaly.anomalType} Anomaly
                  </span>
                  <span className="text-[10px] sm:text-xs text-gray-400">
                    Frames {anomaly.startFrame}-{anomaly.endFrame}
                  </span>
                </div>
                <div className="text-xs sm:text-sm text-gray-300 mb-2">
                  {anomaly.description}
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-600 rounded-full h-1.5 sm:h-2">
                    <div 
                      className="h-1.5 sm:h-2 rounded-full bg-yellow-400"
                      style={{ width: `${anomaly.severity * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] sm:text-xs text-gray-400">
                    {(anomaly.severity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-gray-800 rounded-xl p-4 sm:p-6 border border-gray-700">
        <h3 className="text-lg sm:text-xl font-semibold text-white mb-4">Technical Details</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          <div>
            <h4 className="font-medium text-white mb-2 text-sm sm:text-base">Processing Pipeline</h4>
            <ul className="text-xs sm:text-sm text-gray-300 space-y-1">
              <li>• Frame extraction at 30fps</li>
              <li>• MTCNN face detection</li>
              <li>• Dual-path feature extraction</li>
              <li>• Beadal artifact analysis</li>
              <li>• Temporal sequence modeling</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2 text-sm sm:text-base">Model Architecture</h4>
            <ul className="text-xs sm:text-sm text-gray-300 space-y-1">
              <li>• InceptionResNetV2 (Path A)</li>
              <li>• EfficientNet-B4 (Path B)</li>
              <li>• BiLSTM temporal fusion</li>
              <li>• Attention-based gates</li>
              <li>• Sigmoid classifier</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2 text-sm sm:text-base">Performance Metrics</h4>
            <ul className="text-xs sm:text-sm text-gray-300 space-y-1">
              <li>• {result.frameAnalysis.length} frames analyzed</li>
              <li>• {result.modelMetrics.beadalFeatures} Beadal features</li>
              <li>• {result.tamperLocalization.length} tamper regions</li>
              <li>• {result.temporalAnomalies.length} temporal anomalies</li>
              <li>• {result.processingTime}s processing time</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisDashboard;