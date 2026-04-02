import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { SessionType, mockTracks, Track, sessionTypes } from '../utils/mockData';
import { NowPlaying } from '../components/now-playing';
import { SessionStats } from '../components/session-stats';
import { motion } from 'motion/react';
import { StopCircle, ArrowLeft, Music, Radio } from 'lucide-react';

export function ActiveSession() {
  const location = useLocation();
  const navigate = useNavigate();
  const sessionType = location.state?.sessionType as SessionType;

  const [tracks, setTracks] = useState<Track[]>([]);
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  const sessionTypeInfo = sessionTypes.find(t => t.value === sessionType);

  useEffect(() => {
    if (!sessionType) {
      navigate('/');
      return;
    }

    // Simulate playing tracks
    const trackInterval = setInterval(() => {
      const randomTrack = mockTracks[Math.floor(Math.random() * mockTracks.length)];
      setCurrentTrack(randomTrack);
      setTracks(prev => {
        // Avoid duplicates in recent tracks
        if (prev.length > 0 && prev[prev.length - 1].id === randomTrack.id) {
          return prev;
        }
        return [...prev, randomTrack];
      });
    }, 8000);

    // Set initial track
    setCurrentTrack(mockTracks[0]);
    setTracks([mockTracks[0]]);

    // Session duration timer
    const durationInterval = setInterval(() => {
      setSessionDuration(prev => prev + 1);
    }, 1000);

    return () => {
      clearInterval(trackInterval);
      clearInterval(durationInterval);
    };
  }, [sessionType, navigate]);

  const handleEndSession = () => {
    navigate('/playlists', { 
      state: { 
        newPlaylist: true,
        sessionType,
        tracks 
      } 
    });
  };

  if (!sessionType) return null;

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
          
          <div className="flex items-center gap-4 flex-1">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#1DB954] to-[#1ED760] flex items-center justify-center text-3xl">
              {sessionTypeInfo?.emoji}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Radio className="w-4 h-4 text-[#1DB954]" />
                <span className="text-[#1DB954] text-sm font-semibold uppercase tracking-wider">Live Session</span>
              </div>
              <h1 className="text-4xl font-black text-white">
                {sessionTypeInfo?.label}
              </h1>
            </div>
          </div>

          <button
            onClick={handleEndSession}
            className="px-6 py-3 rounded-full bg-white text-black font-bold hover:scale-105 transition-all flex items-center gap-2"
          >
            <StopCircle className="w-5 h-5" />
            End Session
          </button>
        </motion.div>

        {/* Session Stats */}
        <div className="mb-8">
          <SessionStats tracks={tracks} duration={sessionDuration} />
        </div>

        {/* Now Playing */}
        <div className="mb-8">
          <h2 className="text-white font-bold text-2xl mb-4 flex items-center gap-2">
            Now Playing
          </h2>
          <NowPlaying track={currentTrack} isPlaying={isPlaying} />
        </div>

        {/* Track History */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-white font-bold text-2xl mb-4">Session History</h2>
          <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
            {tracks.slice().reverse().map((track, index) => (
              <motion.div
                key={`${track.id}-${index}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="p-4 rounded-lg bg-[#181818] hover:bg-[#282828] transition-colors flex items-center gap-4 group"
              >
                <div className="text-[#B3B3B3] text-sm w-6 text-right">
                  {tracks.length - index}
                </div>
                <img
                  src={track.imageUrl}
                  alt={track.album}
                  className="w-14 h-14 rounded object-cover"
                />
                <div className="flex-1 min-w-0">
                  <h3 className="text-white font-semibold truncate">{track.name}</h3>
                  <p className="text-[#B3B3B3] text-sm truncate">{track.artist}</p>
                </div>
                <div className="text-sm text-[#B3B3B3]">
                  {Math.floor(track.duration / 60)}:{String(track.duration % 60).padStart(2, '0')}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}