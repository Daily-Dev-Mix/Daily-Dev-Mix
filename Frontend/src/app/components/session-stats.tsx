import { Track } from '../utils/mockData';
import { motion } from 'motion/react';
import { Music, Timer, TrendingUp, Zap } from 'lucide-react';

interface SessionStatsProps {
  tracks: Track[];
  duration: number;
}

export function SessionStats({ tracks, duration }: SessionStatsProps) {
  const avgEnergy = tracks.length > 0
    ? tracks.reduce((sum, t) => sum + t.audioFeatures.energy, 0) / tracks.length
    : 0;
  
  const avgDanceability = tracks.length > 0
    ? tracks.reduce((sum, t) => sum + t.audioFeatures.danceability, 0) / tracks.length
    : 0;

  const stats = [
    {
      icon: Music,
      label: 'Tracks',
      value: tracks.length.toString(),
      color: 'text-[#1DB954]',
    },
    {
      icon: Timer,
      label: 'Duration',
      value: `${Math.floor(duration / 60)}m ${duration % 60}s`,
      color: 'text-blue-400',
    },
    {
      icon: Zap,
      label: 'Energy',
      value: `${Math.round(avgEnergy * 100)}%`,
      color: 'text-yellow-400',
    },
    {
      icon: TrendingUp,
      label: 'Vibe',
      value: `${Math.round(avgDanceability * 100)}%`,
      color: 'text-purple-400',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className="p-5 rounded-lg bg-[#181818]"
        >
          <stat.icon className={`w-6 h-6 mb-3 ${stat.color}`} />
          <div className="text-3xl font-black text-white mb-1">{stat.value}</div>
          <div className="text-sm text-[#B3B3B3]">{stat.label}</div>
        </motion.div>
      ))}
    </div>
  );
}