import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const ThemeContext = createContext({
  dark: false,
  toggle: () => {},
})

export function ThemeProvider({ children }) {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    const stored = localStorage.getItem('skinai-theme')
    if (stored === 'dark' || stored === 'light') return stored === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('skinai-theme', dark ? 'dark' : 'light')
  }, [dark])

  const value = useMemo(
    () => ({
      dark,
      toggle: () => setDark((d) => !d),
      setDark,
    }),
    [dark],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  return useContext(ThemeContext)
}
