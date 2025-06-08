import React from 'react';
import { Shield, Activity, Upload, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

interface HeaderProps {
  currentView: 'upload' | 'analysis' | 'monitor';
  setCurrentView: (view: 'upload' | 'analysis' | 'monitor') => void;
}

const Header: React.FC<HeaderProps> = ({ currentView, setCurrentView }) => {
  const navItems = [
    { id: 'upload', label: 'Upload & Detect', icon: Upload },
    { id: 'analysis', label: 'Analysis', icon: Brain },
    { id: 'monitor', label: 'System Monitor', icon: Activity },
  ];

  return (
    <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Shield className="h-8 w-8 text-cyan-400" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">DeepFake Detector</h1>
              <p className="text-xs text-gray-400">Production AI Security System</p>
            </div>
          </div>

          <nav className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentView === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentView(item.id as any)}
                  className={`relative px-4 py-2 rounded-lg flex items-center space-x-2 transition-all duration-200 ${
                    isActive 
                      ? 'bg-cyan-500 text-white' 
                      : 'text-gray-300 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                  
                  {isActive && (
                    <motion.div
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-cyan-300"
                      layoutId="activeTab"
                      initial={false}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                </button>
              );
            })}
          </nav>

          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-sm font-medium text-white">System Status</div>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-400">Operational</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;