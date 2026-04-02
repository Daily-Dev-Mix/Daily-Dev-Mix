import { Playlist } from '../utils/mockData';
import { motion } from 'motion/react';
import { Music, Clock, Play } from 'lucide-react';

interface PlaylistCardProps {
  playlist: Playlist;
  onClick?: () => void;
}

export function PlaylistCard({ playlist, onClick }: PlaylistCardProps) {
  const totalDuration = playlist.tracks.reduce((sum, track) => sum + track.duration, 0);
  const hours = Math.floor(totalDuration / 3600);
  const minutes = Math.floor((totalDuration % 3600) / 60);

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="group cursor-pointer p-4 rounded-lg bg-[#181818] hover:bg-[#282828] transition-all"
    >
      <div className="relative mb-4">
        <img
          src={playlist.coverImage}
          alt={playlist.name}
          className="w-full aspect-square rounded object-cover shadow-lg"
        />
        <motion.button
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1, scale: 1.1 }}
          className="absolute bottom-2 right-2 w-12 h-12 bg-[#1DB954] rounded-full flex items-center justify-center shadow-xl opacity-0 group-hover:opacity-100 transition-opacity"
        >
          <Play className="w-5 h-5 text-black fill-black ml-0.5" />
        </motion.button>
      </div>

      <h3 className="text-white font-bold text-base mb-2 truncate">{playlist.name}</h3>
      <p className="text-[#B3B3B3] text-sm mb-3 line-clamp-2">{playlist.description}</p>

      <div className="flex items-center gap-3 text-xs text-[#B3B3B3]">
        <div className="flex items-center gap-1">
          <Music className="w-3 h-3" />
          <span>{playlist.tracks.length} songs</span>
        </div>
        <span>•</span>
        <div className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          <span>
            {hours > 0 ? `${hours} hr ` : ''}{minutes} min
          </span>
        </div>
      </div>
    </motion.div>
  );
}