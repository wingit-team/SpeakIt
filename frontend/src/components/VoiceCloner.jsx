import React, { useState } from 'react'
import axios from 'axios'
import './VoiceCloner.css'

const apiBaseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:7860'

function VoiceCloner() {
  const [formData, setFormData] = useState({
    targetText: '',
    referenceText: '',
    referenceAudio: null,
    temperature: 0.3,
    topP: 0.7,
    topK: 20
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [audioName, setAudioName] = useState('')

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('temperature') || name.includes('topP') || name.includes('topK')
        ? parseFloat(value)
        : value
    }))
    setError('')
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setFormData(prev => ({
        ...prev,
        referenceAudio: file
      }))
      setError('')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setAudioUrl('')

    // Validation
    if (!formData.targetText.trim()) {
      setError('Target text is required')
      return
    }
    if (!formData.referenceText.trim()) {
      setError('Reference text is required')
      return
    }
    if (!formData.referenceAudio) {
      setError('Reference audio file is required')
      return
    }

    setLoading(true)
    try {
      const form = new FormData()
      form.append('target_text', formData.targetText)
      form.append('reference_text', formData.referenceText)
      form.append('reference_audio', formData.referenceAudio)
      form.append('temperature', formData.temperature)
      form.append('top_p', formData.topP)
      form.append('top_k', formData.topK)

      const response = await axios.post(`${apiBaseUrl}/clone_voice`, form, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      const url = window.URL.createObjectURL(new Blob([response.data]))
      setAudioUrl(url)
      setAudioName('cloned_voice.wav')
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to clone voice. Please check your inputs.')
      console.error('Voice cloning error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="voice-cloner">
      <h2>🎤 Voice Cloner</h2>
      <p className="subtitle">Clone a voice from reference audio to synthesize new text</p>

      <form onSubmit={handleSubmit} className="cloner-form">
        <div className="form-group">
          <label htmlFor="targetText">Target Text *</label>
          <textarea
            id="targetText"
            name="targetText"
            value={formData.targetText}
            onChange={handleInputChange}
            placeholder="Enter the text you want to synthesize with the cloned voice..."
            rows="4"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="referenceText">Reference Text *</label>
          <textarea
            id="referenceText"
            name="referenceText"
            value={formData.referenceText}
            onChange={handleInputChange}
            placeholder="Transcript of the reference audio file..."
            rows="3"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="referenceAudio">Reference Audio File (.wav, .mp3, etc.) *</label>
          <input
            type="file"
            id="referenceAudio"
            accept="audio/*"
            onChange={handleFileChange}
            disabled={loading}
          />
          {formData.referenceAudio && (
            <p className="file-name">Selected: {formData.referenceAudio.name}</p>
          )}
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="temperature">
              Temperature: {formData.temperature.toFixed(2)}
            </label>
            <input
              type="range"
              id="temperature"
              name="temperature"
              min="0"
              max="1"
              step="0.1"
              value={formData.temperature}
              onChange={handleInputChange}
              disabled={loading}
            />
            <small>Controls randomness (0.0-1.0)</small>
          </div>

          <div className="form-group">
            <label htmlFor="topP">
              Top-P: {formData.topP.toFixed(2)}
            </label>
            <input
              type="range"
              id="topP"
              name="topP"
              min="0"
              max="1"
              step="0.1"
              value={formData.topP}
              onChange={handleInputChange}
              disabled={loading}
            />
            <small>Nucleus sampling</small>
          </div>

          <div className="form-group">
            <label htmlFor="topK">
              Top-K: {formData.topK}
            </label>
            <input
              type="range"
              id="topK"
              name="topK"
              min="1"
              max="100"
              step="1"
              value={formData.topK}
              onChange={handleInputChange}
              disabled={loading}
            />
            <small>Top-k sampling</small>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Cloning Voice...' : 'Clone Voice & Synthesize'}
        </button>
      </form>

      {audioUrl && (
        <div className="audio-output">
          <h3>✅ Synthesized Audio</h3>
          <audio controls>
            <source src={audioUrl} type="audio/wav" />
            Your browser does not support the audio element.
          </audio>
          <a href={audioUrl} download={audioName} className="download-button">
            Download Audio
          </a>
        </div>
      )}
    </div>
  )
}

export default VoiceCloner

