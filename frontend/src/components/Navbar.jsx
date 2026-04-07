import { useTheme } from '../context/ThemeContext'

const Navbar = () => {
  const { dark, toggle } = useTheme()

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/80 bg-white/90 backdrop-blur-md dark:border-slate-800/80 dark:bg-slate-900/90">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-2">
          <span
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 text-sm font-bold text-white shadow-sm"
            aria-hidden
          >
            SA
          </span>
          <div className="text-left">
            <span className="block text-lg font-semibold tracking-tight text-slate-900 dark:text-white">
              SkinAI
            </span>
            <span className="hidden text-xs text-slate-500 dark:text-slate-400 sm:block">
              Clinical decision support
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1 sm:gap-2">
          <nav className="flex items-center gap-1 sm:gap-2" aria-label="Primary">
            <a
              href="#analyze"
              className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
            >
              Analyze
            </a>
            <a
              href="#disclaimer"
              className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
            >
              Disclaimer
            </a>
          </nav>

          <button
            type="button"
            onClick={toggle}
            className="ml-1 flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-slate-700 transition hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-amber-200 dark:hover:bg-slate-700"
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {dark ? (
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path d="M12 3a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0V4a1 1 0 0 0-1-1zm0 15a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm7-8a1 1 0 0 0-1-1h-1a1 1 0 0 0 0 2h1a1 1 0 0 0 1-1zM7 11a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm11.657-4.657a1 1 0 0 0-1.414 0l-.707.707a1 1 0 1 0 1.414 1.414l.707-.707a1 1 0 0 0 0-1.414zM7.05 16.95a1 1 0 0 0-1.414 0 1 1 0 0 0 0 1.414l.707.707a1 1 0 1 0 1.414-1.414l-.707-.707zm-.707-9.193a1 1 0 0 0 0 1.414l.707.707a1 1 0 1 0 1.414-1.414l-.707-.707a1 1 0 0 0-1.414 0zm10.607 10.607a1 1 0 0 0-1.414 0l-.707.707a1 1 0 1 0 1.414 1.414l.707-.707a1 1 0 0 0 0-1.414zM12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z" />
              </svg>
            ) : (
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden>
                <path d="M21.64 13.65A9 9 0 1 1 10.36 2.36a7 7 0 1 0 11.28 11.29z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  )
}

export default Navbar
