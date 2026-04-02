// Mock data for Spotify session tracking

export interface Track {
  id: string;
  name: string;
  artist: string;
  album: string;
  duration: number;
  imageUrl: string;
  audioFeatures: {
    energy: number;
    danceability: number;
    valence: number;
    tempo: number;
    acousticness: number;
  };
}

export interface Session {
  id: string;
  type: SessionType;
  startTime: Date;
  endTime?: Date;
  tracks: Track[];
  isActive: boolean;
}

export interface Playlist {
  id: string;
  name: string;
  description: string;
  tracks: Track[];
  sessionType: SessionType;
  createdAt: Date;
  coverImage: string;
}

export type SessionType = 
  | 'studying' 
  | 'workout' 
  | 'road-trip' 
  | 'relaxing' 
  | 'party' 
  | 'focus';

export const sessionTypes: { value: SessionType; label: string; emoji: string }[] = [
  { value: 'studying', label: 'Studying', emoji: '📚' },
  { value: 'workout', label: 'Workout', emoji: '💪' },
  { value: 'road-trip', label: 'Road Trip', emoji: '🚗' },
  { value: 'relaxing', label: 'Relaxing', emoji: '🧘' },
  { value: 'party', label: 'Party', emoji: '🎉' },
  { value: 'focus', label: 'Deep Focus', emoji: '🎯' },
];

export const mockTracks: Track[] = [
  {
    id: '1',
    name: 'Blinding Lights',
    artist: 'The Weeknd',
    album: 'After Hours',
    duration: 200,
    imageUrl: 'https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=400',
    audioFeatures: {
      energy: 0.73,
      danceability: 0.51,
      valence: 0.33,
      tempo: 171,
      acousticness: 0.001,
    },
  },
  {
    id: '2',
    name: 'Levitating',
    artist: 'Dua Lipa',
    album: 'Future Nostalgia',
    duration: 203,
    imageUrl: 'https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=400',
    audioFeatures: {
      energy: 0.82,
      danceability: 0.70,
      valence: 0.87,
      tempo: 103,
      acousticness: 0.07,
    },
  },
  {
    id: '3',
    name: 'Watermelon Sugar',
    artist: 'Harry Styles',
    album: 'Fine Line',
    duration: 174,
    imageUrl: 'https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400',
    audioFeatures: {
      energy: 0.73,
      danceability: 0.55,
      valence: 0.56,
      tempo: 95,
      acousticness: 0.12,
    },
  },
  {
    id: '4',
    name: 'Sunflower',
    artist: 'Post Malone, Swae Lee',
    album: 'Spider-Man: Into the Spider-Verse',
    duration: 158,
    imageUrl: 'https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=400',
    audioFeatures: {
      energy: 0.48,
      danceability: 0.76,
      valence: 0.91,
      tempo: 90,
      acousticness: 0.56,
    },
  },
  {
    id: '5',
    name: 'Stay',
    artist: 'The Kid LAROI, Justin Bieber',
    album: 'F*CK LOVE 3',
    duration: 141,
    imageUrl: 'https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400',
    audioFeatures: {
      energy: 0.61,
      danceability: 0.59,
      valence: 0.48,
      tempo: 169,
      acousticness: 0.04,
    },
  },
  {
    id: '6',
    name: 'Good 4 U',
    artist: 'Olivia Rodrigo',
    album: 'SOUR',
    duration: 178,
    imageUrl: 'https://images.unsplash.com/photo-1506157786151-b8491531f063?w=400',
    audioFeatures: {
      energy: 0.84,
      danceability: 0.56,
      valence: 0.67,
      tempo: 164,
      acousticness: 0.03,
    },
  },
];

export const mockPlaylists: Playlist[] = [
  {
    id: 'p1',
    name: 'Study Zone Mix',
    description: 'Your personalized study playlist based on focus sessions',
    tracks: mockTracks.slice(0, 4),
    sessionType: 'studying',
    createdAt: new Date('2026-03-10'),
    coverImage: 'https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=400',
  },
  {
    id: 'p2',
    name: 'Workout Energy',
    description: 'High-energy tracks from your best workout sessions',
    tracks: mockTracks.slice(2, 6),
    sessionType: 'workout',
    createdAt: new Date('2026-03-08'),
    coverImage: 'https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=400',
  },
];
