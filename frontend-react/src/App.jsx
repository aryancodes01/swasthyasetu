import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Queue from './pages/Queue'
import Settings from './pages/Settings'
import Editor from './pages/Editor'

const NAV = [
  { label: 'Dashboard', path: '/',         icon: '◈', desc: 'Live pipeline stats' },
  { label: 'Queue',     path: '/queue',     icon: '◷', desc: 'Review & approve'   },
  { label: 'Editor',   path: '/editor',    icon: '✂', desc: 'Preview reels'      },
  { label: 'Settings', path: '/settings',  icon: '◉', desc: 'Configure pipeline' },
]

export default function App() {
  const location = useLocation()

  return (
    <div style={styles.root}>
      {/* Sidebar */}
      <aside style={styles.sidebar}>
        {/* Logo */}
        <div style={styles.logo}>
          <div style={styles.logoIcon}>⚡</div>
          <div>
            <div style={styles.logoName}>ReelForge</div>
            <div style={styles.logoTag}>AI Pipeline</div>
          </div>
        </div>

        {/* Nav */}
        <nav style={styles.nav}>
          {NAV.map(({ label, path, icon, desc }) => {
            const active = location.pathname === path
            return (
              <Link
                key={path}
                to={path}
                style={{
                  ...styles.navItem,
                  ...(active ? styles.navItemActive : {}),
                  textDecoration: 'none',
                }}
              >
                <span style={{ ...styles.navIcon, color: active ? '#22d3ee' : '#475569' }}>
                  {icon}
                </span>
                <div style={styles.navText}>
                  <span style={{ ...styles.navLabel, color: active ? '#e2e8f0' : '#94a3b8' }}>
                    {label}
                  </span>
                  <span style={styles.navDesc}>{desc}</span>
                </div>
                {active && <div style={styles.navActiveDot} />}
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div style={styles.sidebarFooter}>
          <div style={styles.footerStatus}>
            <span style={styles.footerDot} />
            <span style={styles.footerText}>System Online</span>
          </div>
          <div style={styles.footerBranch}>member3/frontend</div>
        </div>
      </aside>

      {/* Main content */}
      <main style={styles.main}>
        <Routes>
          <Route path="/"        element={<Dashboard />} />
          <Route path="/queue"   element={<Queue />}     />
          <Route path="/editor"  element={<Editor />}    />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #030712; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
        input[type=time]::-webkit-calendar-picker-indicator { filter: invert(0.5); cursor: pointer; }
      `}</style>
    </div>
  )
}

const styles = {
  root: {
    display: 'flex',
    minHeight: '100vh',
    background: '#030712',
    fontFamily: "'DM Sans', 'Inter', sans-serif",
  },
  sidebar: {
    width: '220px',
    flexShrink: 0,
    background: 'rgba(8,14,28,0.95)',
    borderRight: '1px solid rgba(255,255,255,0.05)',
    display: 'flex',
    flexDirection: 'column',
    position: 'sticky',
    top: 0,
    height: '100vh',
    overflow: 'hidden',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '1.5rem 1.25rem',
    borderBottom: '1px solid rgba(255,255,255,0.05)',
  },
  logoIcon: {
    fontSize: '22px',
    width: '38px',
    height: '38px',
    background: 'rgba(34,211,238,0.1)',
    border: '1px solid rgba(34,211,238,0.2)',
    borderRadius: '10px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  logoName: {
    fontSize: '15px',
    fontWeight: 700,
    color: '#e2e8f0',
    letterSpacing: '-0.01em',
  },
  logoTag: {
    fontSize: '11px',
    color: '#334155',
    marginTop: '1px',
  },
  nav: {
    flex: 1,
    padding: '1rem 0.75rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  navItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '0.625rem 0.75rem',
    borderRadius: '10px',
    transition: 'background 0.15s',
    position: 'relative',
    border: '1px solid transparent',
  },
  navItemActive: {
    background: 'rgba(34,211,238,0.06)',
    borderColor: 'rgba(34,211,238,0.12)',
  },
  navIcon: {
    fontSize: '16px',
    flexShrink: 0,
    width: '20px',
    textAlign: 'center',
  },
  navText: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1px',
    flex: 1,
    minWidth: 0,
  },
  navLabel: {
    fontSize: '13px',
    fontWeight: 600,
    lineHeight: 1,
  },
  navDesc: {
    fontSize: '11px',
    color: '#334155',
    lineHeight: 1,
  },
  navActiveDot: {
    width: '5px',
    height: '5px',
    borderRadius: '50%',
    background: '#22d3ee',
    flexShrink: 0,
  },
  sidebarFooter: {
    padding: '1rem 1.25rem',
    borderTop: '1px solid rgba(255,255,255,0.05)',
  },
  footerStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    marginBottom: '6px',
  },
  footerDot: {
    display: 'inline-block',
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: '#34d399',
    boxShadow: '0 0 6px #34d399',
  },
  footerText: {
    fontSize: '12px',
    color: '#34d399',
    fontWeight: 500,
  },
  footerBranch: {
    fontSize: '11px',
    color: '#1e293b',
    fontFamily: 'monospace',
  },
  main: {
    flex: 1,
    overflow: 'auto',
    minWidth: 0,
  },
}
