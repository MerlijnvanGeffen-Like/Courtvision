/**
 * API utility functions for communicating with the Flask backend
 */
const API_BASE_URL = 'http://localhost:5000/api'

// Get auth token from localStorage
const getToken = () => {
  return localStorage.getItem('authToken')
}

// Set auth token in localStorage
const setToken = (token) => {
  localStorage.setItem('authToken', token)
}

// Remove auth token from localStorage
const removeToken = () => {
  localStorage.removeItem('authToken')
}

// Make authenticated API request
const apiRequest = async (endpoint, options = {}) => {
  const token = getToken()
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers
  }
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }))
    throw new Error(error.error || `HTTP error! status: ${response.status}`)
  }
  
  return response.json()
}

// Auth API calls
export const authAPI = {
  register: async (username, email, password) => {
    const data = await apiRequest('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password })
    })
    if (data.token) {
      setToken(data.token)
    }
    return data
  },
  
  login: async (username, password) => {
    const data = await apiRequest('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password })
    })
    if (data.token) {
      setToken(data.token)
    }
    return data
  },
  
  logout: () => {
    removeToken()
  },
  
  getCurrentUser: async () => {
    return apiRequest('/auth/me')
  }
}

// Stats API calls
export const statsAPI = {
  getStats: async () => {
    return apiRequest('/stats')
  }
}

// Leaderboard API calls
export const leaderboardAPI = {
  getLeaderboard: async () => {
    return apiRequest('/leaderboard')
  }
}

// Camera API calls
export const cameraAPI = {
  start: async () => {
    return apiRequest('/camera/start', { method: 'POST' })
  },
  
  stop: async () => {
    return apiRequest('/camera/stop', { method: 'POST' })
  },
  
  getStatus: async () => {
    return apiRequest('/camera/status')
  },
  
  reset: async () => {
    return apiRequest('/reset', { method: 'POST' })
  }
}

// Health check API calls (no auth required)
export const healthAPI = {
  check: async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      if (!response.ok) {
        throw new Error('Backend not available')
      }
      return await response.json()
    } catch (error) {
      throw new Error('Backend not available')
    }
  }
}

export { getToken, setToken, removeToken }
