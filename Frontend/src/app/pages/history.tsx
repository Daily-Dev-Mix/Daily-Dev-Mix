import { useNavigate } from 'react-router';
import { sessionTypes, mockTracks, mockPlaylists } from '../utils/mockData';
import { motion } from 'motion/react';
import { ArrowLeft, Calendar, Music, Timer, TrendingUp, Activity } from 'lucide-react';

export function History() {
  const navigate = useNavigate();

  // Mock session history
  const sessions = [
    {
      id: '1',
      type: 'studying' as const,
      startTime: new Date('2026-03-12T09:00:00'),
      endTime: new Date('2026-03-12T11:30:00'),
      trackCount: 23,
      avgEnergy: 0.45,
    },
    {
      id: '2',
      type: 'workout' as const,
      startTime: new Date('2026-03-11T18:00:00'),
      endTime: new Date('2026-03-11T19:15:00'),
      trackCount: 18,
      avgEnergy: 0.87,
    },
    {
      id: '3',
      type: 'relaxing' as const,
      startTime: new Date('2026-03-10T20:00:00'),
      endTime: new Date('2026-03-10T21:45:00'),
      trackCount: 15,
      avgEnergy: 0.32,
    },
    {
      id: '4',
      type: 'road-trip' as const,
      startTime: new Date('2026-03-09T14:00:00'),
      endTime: new Date('2026-03-09T16:30:00'),
      trackCount: 28,
      avgEnergy: 0.68,
    },
  ];

  const getDuration = (start: Date, end: Date) => {
    const diff = end.getTime() - start.getTime();
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    return { hours, minutes };
  };

  const totalListeningTime = sessions.reduce((sum, s) => {
    const dur = getDuration(s.startTime, s.endTime);
    return sum + dur.hours + dur.minutes / 60;
  }, 0);

  return (
    <div className="min-h-screen bg-black p-6 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-6 mb-8"
        >
          <button
            onClick={() => navigate('/')}
            className="p-3 rounded-full bg-black/40 hover:bg-[#282828] transition-all text-white border border-[#282828]"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          
          <div>
            <h1 className="text-5xl md:text-6xl font-black text-white mb-2">
              Listening History
            </h1>
            <p className="text-[#B3B3B3] text-lg">
              Your past sessions and insights
            </p>
          </div>
        </motion.div>

        {/* Stats Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12"
        >
          <div className="p-6 rounded-lg bg-gradient-to-br from-purple-500/10 to-transparent border border-[#282828]">
            <Activity className="w-8 h-8 text-purple-400 mb-3" />
            <div className="text-4xl font-black text-white mb-1">
              {sessions.length}
            </div>
            <div className="text-[#B3B3B3] text-sm">Total Sessions</div>
          </div>
          <div className="p-6 rounded-lg bg-gradient-to-br from-[#1DB954]/10 to-transparent border border-[#282828]">
            <Music className="w-8 h-8 text-[#1DB954] mb-3" />
            <div className="text-4xl font-black text-white mb-1">
              {sessions.reduce((sum, s) => sum + s.trackCount, 0)}
            </div>
            <div className="text-[#B3B3B3] text-sm">Tracks Played</div>
          </div>
          <div className="p-6 rounded-lg bg-gradient-to-br from-blue-500/10 to-transparent border border-[#282828]">
            <Timer className="w-8 h-8 text-blue-400 mb-3" />
            <div className="text-4xl font-black text-white mb-1">
              {Math.round(totalListeningTime)}h
            </div>
            <div className="text-[#B3B3B3] text-sm">Listening Time</div>
          </div>
          <div className="p-6 rounded-lg bg-gradient-to-br from-pink-500/10 to-transparent border border-[#282828]">
            <TrendingUp className="w-8 h-8 text-pink-400 mb-3" />
            <div className="text-4xl font-black text-white mb-1">
              {mockPlaylists.length}
            </div>
            <div className="text-[#B3B3B3] text-sm">Playlists Created</div>
          </div>
        </motion.div>

        {/* Sessions List */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-2xl font-bold text-white mb-6">Recent Sessions</h2>
          <div className="space-y-4">
            {sessions.map((session, index) => {
              const sessionTypeInfo = sessionTypes.find(t => t.value === session.type);
              const duration = getDuration(session.startTime, session.endTime);

              return (
                <motion.div
                  key={session.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.05 }}
                  className="p-6 rounded-lg bg-[#181818] hover:bg-[#282828] transition-all group cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-[#1DB954]/20 to-[#1DB954]/5 flex items-center justify-center text-3xl border border-[#1DB954]/20">
                        {sessionTypeInfo?.emoji}
                      </div>
                      <div>
                        <h3 className="text-white font-bold text-xl">
                          {sessionTypeInfo?.label}
                        </h3>
                        <div className="flex items-center gap-3 text-[#B3B3B3] text-sm mt-1">
                          <span className="flex items-center gap-1">
                            <Calendar className="w-4 h-4" />
                            {session.startTime.toLocaleDateString('en-US', { 
                              month: 'short', 
                              day: 'numeric',
                              year: 'numeric'
                            })}
                          </span>
                          <span>•</span>
                          <span>
                            {session.startTime.toLocaleTimeString('en-US', { 
                              hour: '2-digit', 
                              minute: '2-digit' 
                            })} - {session.endTime.toLocaleTimeString('en-US', { 
                              hour: '2-digit', 
                              minute: '2-digit' 
                            })}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-4 rounded-lg bg-black/30">
                      <Music className="w-5 h-5 mx-auto mb-2 text-[#1DB954]" />
                      <div className="text-white font-bold text-lg">{session.trackCount}</div>
                      <div className="text-[#B3B3B3] text-xs">Tracks</div>
                    </div>
                    <div className="text-center p-4 rounded-lg bg-black/30">
                      <Timer className="w-5 h-5 mx-auto mb-2 text-blue-400" />
                      <div className="text-white font-bold text-lg">
                        {duration.hours}h {duration.minutes}m
                      </div>
                      <div className="text-[#B3B3B3] text-xs">Duration</div>
                    </div>
                    <div className="text-center p-4 rounded-lg bg-black/30">
                      <TrendingUp className="w-5 h-5 mx-auto mb-2 text-yellow-400" />
                      <div className="text-white font-bold text-lg">
                        {Math.round(session.avgEnergy * 100)}%
                      </div>
                      <div className="text-[#B3B3B3] text-xs">Avg Energy</div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </div>
  );
}