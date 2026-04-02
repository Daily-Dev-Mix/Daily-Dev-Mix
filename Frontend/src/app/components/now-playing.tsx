import { Track } from '../utils/mockData';
import { motion } from 'motion/react';
import { Music2, Pause, Play } from 'lucide-react';

interface NowPlayingProps {
  track: Track | null;
  isPlaying: boolean;
}

export function NowPlaying({ track, isPlaying }: NowPlayingProps) {
  if (!track) {
    return (
      <div className="p-6 rounded-lg bg-[#181818] text-center">
        <Music2 className="w-12 h-12 mx-auto mb-4 text-[#535353]" />
        <p className="text-[#B3B3B3]">No track playing</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="p-6 rounded-lg bg-[#181818] hover:bg-[#282828] transition-colors"
    >
      <div className="flex items-center gap-4">
        <div className="relative">
          <img
            src={track.imageUrl}
            alt={track.album}
            className="w-24 h-24 rounded object-cover shadow-lg"
          />
          {isPlaying && (
            <motion.div
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="absolute inset-0 rounded bg-[#1DB954]/20 flex items-center justify-center"
            >
              <div className="w-3 h-3 bg-[#1DB954] rounded-full animate-pulse" />
            </motion.div>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-white font-bold text-xl truncate">{track.name}</h3>
          <p className="text-[#B3B3B3] truncate">{track.artist}</p>
          <p className="text-[#535353] text-sm truncate">{track.album}</p>
        </div>
      </div>
      
      <div className="mt-6 grid grid-cols-5 gap-3 text-xs">
        <div className="text-center">
          <div className="text-[#B3B3B3] mb-1">Energy</div>
          <div className="h-1 bg-[#282828] rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#1DB954]" 
              style={{ width: `${track.audioFeatures.energy * 100}%` }}
            />
          </div>
          <div className="text-white font-semibold mt-1">{Math.round(track.audioFeatures.energy * 100)}%</div>
        </div>
        <div className="text-center">
          <div className="text-[#B3B3B3] mb-1">Dance</div>
          <div className="h-1 bg-[#282828] rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#1DB954]" 
              style={{ width: `${track.audioFeatures.danceability * 100}%` }}
            />
          </div>
          <div className="text-white font-semibold mt-1">{Math.round(track.audioFeatures.danceability * 100)}%</div>
        </div>
        <div className="text-center">
          <div className="text-[#B3B3B3] mb-1">Mood</div>
          <div className="h-1 bg-[#282828] rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#1DB954]" 
              style={{ width: `${track.audioFeatures.valence * 100}%` }}
            />
          </div>
          <div className="text-white font-semibold mt-1">{Math.round(track.audioFeatures.valence * 100)}%</div>
        </div>
        <div className="text-center">
          <div className="text-[#B3B3B3] mb-1">Tempo</div>
          <div className="text-white font-semibold">{Math.round(track.audioFeatures.tempo)}</div>
        </div>
        <div className="text-center">
          <div className="text-[#B3B3B3] mb-1">Acoustic</div>
          <div className="h-1 bg-[#282828] rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#1DB954]" 
              style={{ width: `${track.audioFeatures.acousticness * 100}%` }}
            />
          </div>
          <div className="text-white font-semibold mt-1">{Math.round(track.audioFeatures.acousticness * 100)}%</div>
        </div>
      </div>
    </motion.div>
  );
}