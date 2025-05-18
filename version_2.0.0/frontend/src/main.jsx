import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { UserProvider } from './context/UserContext'
import { SubscriptionProvider } from './context/SubscriptionContext'
import { BrowserRouter } from 'react-router-dom'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <UserProvider>
        <SubscriptionProvider>
          <App />
        </SubscriptionProvider>
      </UserProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
