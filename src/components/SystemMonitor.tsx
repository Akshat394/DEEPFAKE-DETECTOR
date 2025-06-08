import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, HardDrive, Users, Clock, AlertTriangle, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { SystemStatus } from '../types';

interface SystemMetrics {
  gpuUtilization: number;
  memoryUsage: number;
  processingQueue: number;
  averageLatency: number;
}

interface SystemMonitorProps {
  metrics: SystemMetrics;
}

const SystemMonitor: React.FC<SystemMonitorProps> = ({ metrics }) => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    gpuUtilization: 78,
    queueLength: 12,
    processedVideos: 2847,
    averageLatency: 0.42,
    modelAccuracy: 96.3,
    systemHealth: 'healthy',
  });

  const [performanceData, setPerformanceData] = useState(
    Array.from({ length: 20 }, (_, i) => ({
      time: i,
      gpu: 70 + Math.random() * 20,
      memory: 60 + Math.random() * 25,
      throughput: 85 + Math.random() * 15,
    }))
  );

  useEffect(() => {
    const interval = setInterval(() => {
      setPerformanceData(prev => {
        const newData = [...prev.slice(1), {
          time: prev[prev.length - 1].time + 1,
          gpu: 70 + Math.random() * 20,
          memory: 60 + Math.random() * 25,
          throughput: 85 + Math.random() * 15,
        }];
        return newData;
      });

      setSystemStatus(prev => ({
        ...prev,
        gpuUtilization: 70 + Math.random() * 20,
        queueLength: Math.max(0, prev.queueLength + Math.floor(Math.random() * 5) - 2),
        processedVideos: prev.processedVideos + Math.floor(Math.random() * 3),
        averageLatency: 0.3 + Math.random() * 0.3,
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const queueData = [
    { name: 'Processing', value: 8, color: '#06B6D4' },
    { name: 'Queued', value: systemStatus.queueLength - 8, color: '#F59E0B' },
    { name: 'Available', value: 50 - systemStatus.queueLength, color: '#10B981' },
  ];

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-400 bg-green-400/20';
      case 'warning': return 'text-yellow-400 bg-yellow-400/20';
      case 'critical': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getStatusColor = (value: number) => {
    if (value >= 80) return 'text-red-500';
    if (value >= 60) return 'text-yellow-500';
    return 'text-green-500';
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">System Status</h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* GPU Utilization */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Cpu className="h-5 w-5 text-blue-500" />
                <span className="text-sm font-medium text-gray-700">GPU Usage</span>
              </div>
              <span className={`text-sm font-semibold ${getStatusColor(metrics.gpuUtilization)}`}>
                {metrics.gpuUtilization}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${metrics.gpuUtilization}%` }}
              />
            </div>
          </div>

          {/* Memory Usage */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <HardDrive className="h-5 w-5 text-purple-500" />
                <span className="text-sm font-medium text-gray-700">Memory</span>
              </div>
              <span className={`text-sm font-semibold ${getStatusColor(metrics.memoryUsage)}`}>
                {metrics.memoryUsage}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${metrics.memoryUsage}%` }}
              />
            </div>
          </div>

          {/* Processing Queue */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-green-500" />
                <span className="text-sm font-medium text-gray-700">Queue</span>
              </div>
              <span className="text-sm font-semibold text-gray-900">
                {metrics.processingQueue} items
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(metrics.processingQueue / 10) * 100}%` }}
              />
            </div>
          </div>

          {/* Average Latency */}
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Zap className="h-5 w-5 text-yellow-500" />
                <span className="text-sm font-medium text-gray-700">Latency</span>
              </div>
              <span className="text-sm font-semibold text-gray-900">
                {metrics.averageLatency}ms
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(metrics.averageLatency / 1000) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemMonitor;