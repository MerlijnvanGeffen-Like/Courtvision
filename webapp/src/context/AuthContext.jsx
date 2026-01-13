import { createContext, useContext, useState, useEffect } from 'react'
import { authAPI, getToken } from '../utils/api'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Check if user is logged in on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = getToken()
      if (token) {
        try {
          const response = await authAPI.getCurrentUser()
          if (response.user) {
            setUser(response.user)
          }
        } catch (err) {
          // Token is invalid, remove it
          authAPI.logout()
        }
      }
      setLoading(false)
    }
    checkAuth()
  }, [])

  const login = async (username, password) => {
    try {
      setError(null)
      const response = await authAPI.login(username, password)
      if (response.user) {
        setUser(response.user)
        return { success: true }
      }
      return { success: false, error: 'Login failed' }
    } catch (err) {
      setError(err.message)
      return { success: false, error: err.message }
    }
  }

  const register = async (username, email, password) => {
    try {
      setError(null)
      const response = await authAPI.register(username, email, password)
      if (response.user) {
        setUser(response.user)
        return { success: true }
      }
      return { success: false, error: 'Registration failed' }
    } catch (err) {
      setError(err.message)
      return { success: false, error: err.message }
    }
  }

  const logout = () => {
    authAPI.logout()
    setUser(null)
    setError(null)
  }

  const value = {
    user,
    loading,
    error,
    login,
    register,
    logout,
    isAuthenticated: !!user
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
