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
    <div className="w-full max-w-4xl mx-auto p-4 space-y-6">
      {/* Overall Result Card */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col sm:flex-row items-center justify-between">
          <div className="flex items-center space-x-3 mb-4 sm:mb-0">
            {result.isFake ? (
              <AlertTriangle className="h-8 w-8 text-red-500" />
            ) : (
              <CheckCircle className="h-8 w-8 text-green-500" />
            )}
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                {result.isFake ? 'Deepfake Detected' : 'Authentic Video'}
              </h2>
              <p className="text-sm text-gray-500">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-500">Processing Time</p>
            <p className="text-lg font-semibold text-gray-900">{result.processingTime.toFixed(1)}s</p>
          </div>
        </div>
      </div>

      {/* Frame Analysis Chart */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Frame-by-Frame Analysis</h3>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={frameData}>
              <XAxis dataKey="frame" />
              <YAxis domain={[0, 100]} />
              <Tooltip
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Confidence']}
                labelFormatter={(label) => `Frame ${label}`}
              />
              <Bar
                dataKey="confidence"
                fill={result.isFake ? '#ef4444' : '#22c55e'}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Video Metadata */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Video Information</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500">Duration</p>
            <p className="text-lg font-semibold text-gray-900">{result.metadata.videoDuration}s</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Frames</p>
            <p className="text-lg font-semibold text-gray-900">{result.metadata.frameCount}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Resolution</p>
            <p className="text-lg font-semibold text-gray-900">{result.metadata.resolution}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Format</p>
            <p className="text-lg font-semibold text-gray-900">{result.metadata.format}</p>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Model Performance</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={modelMetricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="name" 
                stroke="#9CA3AF"
                fontSize={11}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                stroke="#9CA3AF"
                fontSize={12}
                domain={[0, 100]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#fff'
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

      {/* ASCII Preview & Temporal Anomalies */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ASCII Preview */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <FileText className="h-5 w-5 text-green-400" />
            <h3 className="text-xl font-semibold text-white">ASCII Pattern Preview</h3>
          </div>
          <div className="bg-black rounded-lg p-4 font-mono text-xs">
            {result.asciiPreview.slice(0, 3).map((line, idx) => (
              <div key={idx} className="text-green-400 leading-tight">
                {line}
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm text-gray-400">
            Novel ASCII conversion reduces compute by {result.modelMetrics.computeReduction}%
          </div>
        </div>

        {/* Temporal Anomalies */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Eye className="h-5 w-5 text-yellow-400" />
            <h3 className="text-xl font-semibold text-white">Temporal Anomalies</h3>
          </div>
          <div className="space-y-3">
            {result.temporalAnomalies.map((anomaly, idx) => (
              <div key={idx} className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-white capitalize">
                    {anomaly.anomalType} Anomaly
                  </span>
                  <span className="text-xs text-gray-400">
                    Frames {anomaly.startFrame}-{anomaly.endFrame}
                  </span>
                </div>
                <div className="text-sm text-gray-300 mb-2">
                  {anomaly.description}
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-600 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full bg-yellow-400"
                      style={{ width: `${anomaly.severity * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-400">
                    {(anomaly.severity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Technical Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium text-white mb-2">Processing Pipeline</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Frame extraction at 30fps</li>
              <li>• MTCNN face detection</li>
              <li>• Dual-path feature extraction</li>
              <li>• Beadal artifact analysis</li>
              <li>• Temporal sequence modeling</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Model Architecture</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• InceptionResNetV2 (Path A)</li>
              <li>• EfficientNet-B4 (Path B)</li>
              <li>• BiLSTM temporal fusion</li>
              <li>• Attention-based gates</li>
              <li>• Sigmoid classifier</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Performance Metrics</h4>
            <ul className="text-sm text-gray-300 space-y-1">
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