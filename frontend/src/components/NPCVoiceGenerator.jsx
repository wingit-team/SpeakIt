import React, { useState } from 'react'
import axios from 'axios'
import './NPCVoiceGenerator.css'

const apiBaseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:7860'

function NPCVoiceGenerator() {
  const [text, setText] = useState('')
  const [npcId, setNpcId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [resolvedNpcId, setResolvedNpcId] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setAudioUrl('')
    setResolvedNpcId('')

    if (!text.trim()) {
      setError('Text is required')
      return
    }

    setLoading(true)
    try {
      const payload = {
        text: text.trim(),
        npc_id: npcId.trim() ? npcId.trim() : null
      }

      const response = await axios.post(`${apiBaseUrl}/generate_npc_speech`, payload, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      const url = window.URL.createObjectURL(new Blob([response.data]))
      setAudioUrl(url)
      setResolvedNpcId(response.headers['x-npc-id'] || payload.npc_id || '')
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate NPC speech. Please try again.')
      console.error('NPC speech error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="npc-generator">
      <h2>🧠 NPC Voice Generator</h2>
      <p className="subtitle">Generate procedural NPC voices with persistent latents</p>

      <form onSubmit={handleSubmit} className="npc-form">
        <div className="form-group">
          <label htmlFor="npcText">Text *</label>
          <textarea
            id="npcText"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter the NPC dialogue you want to synthesize..."
            rows="4"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="npcId">NPC ID (optional)</label>
          <input
            type="text"
            id="npcId"
            value={npcId}
            onChange={(e) => setNpcId(e.target.value)}
            placeholder="Leave blank to auto-generate"
            disabled={loading}
          />
          <small>If left empty, a new NPC ID will be generated</small>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Generating Speech...' : 'Generate NPC Speech'}
        </button>
      </form>

      {audioUrl && (
        <div className="audio-output">
          <h3>✅ Generated Audio</h3>
          {resolvedNpcId && (
            <p className="npc-id">NPC ID: <strong>{resolvedNpcId}</strong></p>
          )}
          <audio controls>
            <source src={audioUrl} type="audio/wav" />
            Your browser does not support the audio element.
          </audio>
          <a href={audioUrl} download="npc_speech.wav" className="download-button">
            Download Audio
          </a>
        </div>
      )}
    </div>
  )
}

export default NPCVoiceGenerator

