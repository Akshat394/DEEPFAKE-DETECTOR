import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, HardDrive, Users, Clock, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { SystemStatus } from '../types';

const SystemMonitor: React.FC = () => {
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

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* System Status Header */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">System Monitor</h2>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
            getHealthColor(systemStatus.systemHealth)
          }`}>
            <Activity className="h-4 w-4" />
            <span className="text-sm font-medium capitalize">{systemStatus.systemHealth}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-cyan-400 mb-1">
              {systemStatus.gpuUtilization.toFixed(0)}%
            </div>
            <div className="text-sm text-gray-400 flex items-center justify-center space-x-1">
              <Cpu className="h-3 w-3" />
              <span>GPU Usage</span>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">
              {systemStatus.queueLength}
            </div>
            <div className="text-sm text-gray-400 flex items-center justify-center space-x-1">
              <Clock className="h-3 w-3" />
              <span>Queue Length</span>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">
              {systemStatus.processedVideos.toLocaleString()}
            </div>
            <div className="text-sm text-gray-400 flex items-center justify-center space-x-1">
              <Users className="h-3 w-3" />
              <span>Processed</span>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">
              {systemStatus.averageLatency.toFixed(2)}s
            </div>
            <div className="text-sm text-gray-400 flex items-center justify-center space-x-1">
              <Activity className="h-3 w-3" />
              <span>Avg Latency</span>
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400 mb-1">
              {systemStatus.modelAccuracy.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400 flex items-center justify-center space-x-1">
              <HardDrive className="h-3 w-3" />
              <span>Accuracy</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Performance */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-4">Real-time Performance</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9CA3AF"
                  fontSize={12}
                  domain={['dataMin', 'dataMax']}
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
                <Line 
                  type="monotone" 
                  dataKey="gpu" 
                  stroke="#06B6D4" 
                  strokeWidth={2}
                  dot={false}
                  name="GPU Utilization"
                />
                <Line 
                  type="monotone" 
                  dataKey="memory" 
                  stroke="#F59E0B" 
                  strokeWidth={2}
                  dot={false}
                  name="Memory Usage"
                />
                <Line 
                  type="monotone" 
                  dataKey="throughput" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  dot={false}
                  name="Throughput"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Queue Status */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-4">Processing Queue</h3>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={queueData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {queueData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#fff'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4">
            {queueData.map((item, idx) => (
              <div key={idx} className="text-center">
                <div className="flex items-center justify-center space-x-2 mb-1">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-400">{item.name}</span>
                </div>
                <div className="text-lg font-semibold text-white">{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Performance Metrics */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Model Performance Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-4">
            <h4 className="font-medium text-white">Detection Accuracy</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">FaceForensics++ (c23)</span>
                <span className="text-white">96.8%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: '96.8%' }}></div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Celeb-DF</span>
                <span className="text-white">94.2%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: '94.2%' }}></div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">DFDC Preview</span>
                <span className="text-white">91.5%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-400 h-2 rounded-full" style={{ width: '91.5%' }}></div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="font-medium text-white">Processing Speed</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Frames/Second</span>
                <span className="text-cyan-400 font-medium">2.4 fps</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Videos/Hour</span>
                <span className="text-cyan-400 font-medium">127</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Avg Response</span>
                <span className="text-cyan-400 font-medium">{systemStatus.averageLatency.toFixed(2)}s</span>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="font-medium text-white">System Health</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Uptime</span>
                <span className="text-green-400 font-medium">99.7%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Error Rate</span>
                <span className="text-green-400 font-medium">0.03%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Model Drift</span>
                <span className="text-yellow-400 font-medium">Low</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alerts & Notifications */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-white mb-4">Recent Alerts</h3>
        <div className="space-y-3">
          <div className="flex items-start space-x-3 p-3 bg-yellow-400/10 border border-yellow-400/20 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="text-sm font-medium text-white">Model Accuracy Drift Detected</div>
              <div className="text-xs text-gray-400 mt-1">
                Accuracy dropped 0.3% over the last 24 hours. Consider retraining schedule.
              </div>
              <div className="text-xs text-yellow-400 mt-1">2 hours ago</div>
            </div>
          </div>
          
          <div className="flex items-start space-x-3 p-3 bg-blue-400/10 border border-blue-400/20 rounded-lg">
            <Activity className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="text-sm font-medium text-white">High Processing Load</div>
              <div className="text-xs text-gray-400 mt-1">
                Queue length exceeded threshold. Auto-scaling initiated.
              </div>
              <div className="text-xs text-blue-400 mt-1">1 hour ago</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemMonitor;