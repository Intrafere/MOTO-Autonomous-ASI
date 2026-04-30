import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import { installNamespacedLocalStorage } from './utils/runtimeConfig'

installNamespacedLocalStorage()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

