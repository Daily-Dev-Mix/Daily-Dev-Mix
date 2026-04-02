import { useState } from 'react';
import { useNavigate } from 'react-router';
import { SessionTypeSelector } from '../components/session-type-selector';
import { SessionType } from '../utils/mockData';
import { motion } from 'motion/react';
import { Play, History, ListMusic, Sparkles } from 'lucide-react';

export function Home() {
  const navigate = useNavigate();
  const [selectedType, setSelectedType] = useState<SessionType | null>(null);

  const handleStartSession = () => {
    if (selectedType) {
      navigate('/session', { state: { sessionType: selectedType } });
    }
  };

  return (
    <div className="min-h-screen bg-black p-6 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-5xl md:text-7xl font-black text-white mb-4">
            Daily Dev Mix
          </h1>
          <p className="text-xl text-[#B3B3B3] max-w-3xl">
            Track your listening sessions and get truly personalized playlists based on your actual music preferences
          </p>
        </motion.div>

        {/* Session Type Selection */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-6">
            Choose your vibe
          </h2>
          <SessionTypeSelector
            selectedType={selectedType}
            onSelectType={setSelectedType}
          />
        </motion.div>

        {/* Start Session Button */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-16"
        >
          <button
            onClick={handleStartSession}
            disabled={!selectedType}
            className={`
              px-10 py-4 rounded-full text-base font-bold transition-all
              inline-flex items-center gap-3
              ${selectedType
                ? 'bg-[#1DB954] text-black hover:bg-[#1ED760] hover:scale-105'
                : 'bg-[#282828] text-[#535353] cursor-not-allowed'
              }
            `}
          >
            <Play className="w-5 h-5 fill-current" />
            Start Listening Session
          </button>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h2 className="text-2xl font-bold text-white mb-6">Quick access</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <button
              onClick={() => navigate('/playlists')}
              className="p-6 rounded-lg bg-[#181818] hover:bg-[#282828] transition-all text-left group"
            >
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <ListMusic className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h3 className="text-white font-bold text-lg mb-1">
                    Your Playlists
                  </h3>
                  <p className="text-[#B3B3B3] text-sm">
                    Browse your personalized mixes
                  </p>
                </div>
              </div>
            </button>

            <button
              onClick={() => navigate('/history')}
              className="p-6 rounded-lg bg-[#181818] hover:bg-[#282828] transition-all text-left group"
            >
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <History className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h3 className="text-white font-bold text-lg mb-1">
                    Listening History
                  </h3>
                  <p className="text-[#B3B3B3] text-sm">
                    View past sessions and insights
                  </p>
                </div>
              </div>
            </button>
          </div>
        </motion.div>

        {/* Info Box */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-16 p-8 rounded-lg bg-gradient-to-r from-[#1DB954]/10 to-[#1DB954]/5 border border-[#1DB954]/20"
        >
          <h3 className="text-white font-bold text-xl mb-4 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-[#1DB954]" />
            How it works
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <div className="flex gap-3 mb-4">
                <div className="w-8 h-8 rounded-full bg-[#1DB954] flex items-center justify-center flex-shrink-0 font-bold text-black">
                  1
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Choose Activity</h4>
                  <p className="text-[#B3B3B3] text-sm">Select what you're doing - studying, working out, or chilling</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-[#1DB954] flex items-center justify-center flex-shrink-0 font-bold text-black">
                  2
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Track Metadata</h4>
                  <p className="text-[#B3B3B3] text-sm">We analyze audio features and listening patterns</p>
                </div>
              </div>
            </div>
            <div>
              <div className="flex gap-3 mb-4">
                <div className="w-8 h-8 rounded-full bg-[#1DB954] flex items-center justify-center flex-shrink-0 font-bold text-black">
                  3
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Generate Playlists</h4>
                  <p className="text-[#B3B3B3] text-sm">AI creates personalized mixes based on your taste</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-[#1DB954] flex items-center justify-center flex-shrink-0 font-bold text-black">
                  4
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Enjoy Your Mix</h4>
                  <p className="text-[#B3B3B3] text-sm">Music that truly matches your vibe and activity</p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}