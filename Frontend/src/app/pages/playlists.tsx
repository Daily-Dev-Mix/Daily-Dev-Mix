import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router';
import { PlaylistCard } from '../components/playlist-card';
import { mockPlaylists, Playlist, sessionTypes } from '../utils/mockData';
import { motion } from 'motion/react';
import { ArrowLeft, Sparkles, Music, CheckCircle } from 'lucide-react';

export function Playlists() {
  const navigate = useNavigate();
  const location = useLocation();
  const [playlists, setPlaylists] = useState<Playlist[]>(mockPlaylists);
  const [showNewPlaylist, setShowNewPlaylist] = useState(false);

  useEffect(() => {
    // Check if we're coming from a completed session
    if (location.state?.newPlaylist) {
      const { sessionType, tracks } = location.state;
      const sessionTypeInfo = sessionTypes.find(t => t.value === sessionType);
      
      const newPlaylist: Playlist = {
        id: `p${Date.now()}`,
        name: `${sessionTypeInfo?.label} Mix ${new Date().toLocaleDateString()}`,
        description: `Personalized playlist from your ${sessionTypeInfo?.label.toLowerCase()} session`,
        tracks: tracks || [],
        sessionType,
        createdAt: new Date(),
        coverImage: tracks?.[0]?.imageUrl || 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400',
      };

      setPlaylists(prev => [newPlaylist, ...prev]);
      setShowNewPlaylist(true);

      // Clear the state
      window.history.replaceState({}, document.title);

      // Auto-hide the new playlist banner
      setTimeout(() => setShowNewPlaylist(false), 5000);
    }
  }, [location]);

  return (
    <div className="min-h-screen bg-black p-6 md:p-8">
      <div className="max-w-7xl mx-auto">
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
              Your Library
            </h1>
            <p className="text-[#B3B3B3] text-lg">
              Personalized playlists generated from your sessions
            </p>
          </div>
        </motion.div>

        {/* New Playlist Alert */}
        {showNewPlaylist && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-8 p-5 rounded-lg bg-[#1DB954]/10 border border-[#1DB954]/30"
          >
            <div className="flex items-center gap-3">
              <CheckCircle className="w-6 h-6 text-[#1DB954]" />
              <div>
                <h3 className="text-white font-bold text-lg">
                  Playlist Generated Successfully!
                </h3>
                <p className="text-[#B3B3B3] text-sm">
                  Your personalized mix is ready based on your session
                </p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Playlists Grid */}
        {playlists.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-24"
          >
            <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-[#282828] flex items-center justify-center">
              <Music className="w-12 h-12 text-[#535353]" />
            </div>
            <h2 className="text-3xl font-black text-white mb-3">
              No playlists yet
            </h2>
            <p className="text-[#B3B3B3] text-lg mb-8 max-w-md mx-auto">
              Start a listening session to generate your first personalized playlist
            </p>
            <button
              onClick={() => navigate('/')}
              className="px-8 py-4 rounded-full bg-[#1DB954] text-black font-bold hover:bg-[#1ED760] hover:scale-105 transition-all"
            >
              Start Your First Session
            </button>
          </motion.div>
        ) : (
          <>
            <div className="grid md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
              {playlists.map((playlist, index) => (
                <motion.div
                  key={playlist.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <PlaylistCard
                    playlist={playlist}
                    onClick={() => {
                      // In a real app, this would navigate to playlist details
                      console.log('View playlist:', playlist.name);
                    }}
                  />
                </motion.div>
              ))}
            </div>

            {/* Info Section */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="mt-16 p-8 rounded-lg bg-gradient-to-r from-[#1DB954]/5 to-transparent border border-[#282828]"
            >
              <div className="flex items-start gap-4">
                <Sparkles className="w-8 h-8 text-[#1DB954] flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-white font-bold text-xl mb-3">
                    AI-Powered Personalization
                  </h3>
                  <p className="text-[#B3B3B3] leading-relaxed">
                    Each playlist is generated using advanced audio analysis and machine learning. 
                    We analyze energy levels, danceability, mood (valence), tempo, and acoustic features 
                    from the tracks you listened to during your session. Our algorithm then finds similar 
                    tracks that match your unique taste - creating truly personalized mixes, not just 
                    generic activity-based recommendations.
                  </p>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </div>
    </div>
  );
}