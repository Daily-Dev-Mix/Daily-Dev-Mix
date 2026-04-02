import { createBrowserRouter } from 'react-router';
import { Home } from './pages/home';
import { ActiveSession } from './pages/active-session';
import { Playlists } from './pages/playlists';
import { History } from './pages/history';

export const router = createBrowserRouter([
  {
    path: '/',
    Component: Home,
  },
  {
    path: '/session',
    Component: ActiveSession,
  },
  {
    path: '/playlists',
    Component: Playlists,
  },
  {
    path: '/history',
    Component: History,
  },
]);
