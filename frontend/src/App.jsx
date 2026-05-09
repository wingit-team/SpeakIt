import React, { useState } from 'react'
import VoiceCloner from './components/VoiceCloner'
import NPCVoiceGenerator from './components/NPCVoiceGenerator'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('cloner')

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>SpeakIt</h1>
        <p>Headless TTS Microservice - Voice Cloning & NPC Voice Generation</p>
      </header>

      <nav className="app-nav">
        <button
          className={`nav-button ${activeTab === 'cloner' ? 'active' : ''}`}
          onClick={() => setActiveTab('cloner')}
        >
          Voice Cloner
        </button>
        <button
          className={`nav-button ${activeTab === 'npc' ? 'active' : ''}`}
          onClick={() => setActiveTab('npc')}
        >
          NPC Voice Generator
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'cloner' && <VoiceCloner />}
        {activeTab === 'npc' && <NPCVoiceGenerator />}
      </main>
    </div>
  )
}

export default App
