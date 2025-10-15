import React, { useEffect, useMemo, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

function useWebcamCapture(intervalMs: number, onFrame: (blob: Blob) => void) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    let timer: number | null = null
    let stopped = false

    async function start() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: false })
      const video = document.createElement('video')
      video.autoplay = true
      video.playsInline = true
      video.srcObject = stream
      videoRef.current = video

      const canvas = document.createElement('canvas')
      canvas.width = 640
      canvas.height = 360
      canvasRef.current = canvas

      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => resolve()
      })

      const ctx = canvas.getContext('2d')!

      const tick = () => {
        if (stopped) return
        if (video.readyState >= 2) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          canvas.toBlob((blob) => {
            if (blob) onFrame(blob)
          }, 'image/jpeg', 0.85)
        }
        timer = window.setTimeout(tick, intervalMs)
      }
      tick()
    }

    start().catch(console.error)
    return () => {
      stopped = true
      if (timer) window.clearTimeout(timer)
      const v = videoRef.current
      if (v && v.srcObject) {
        (v.srcObject as MediaStream).getTracks().forEach(t => t.stop())
      }
    }
  }, [intervalMs, onFrame])
}

export function App() {
  const [focusPercent, setFocusPercent] = useState<number>(100)
  const [events, setEvents] = useState<string[]>([])
  const [history30, setHistory30] = useState<{ time: number, focusPercent: number }[]>([])
  const [history7, setHistory7] = useState<{ time: number, focusPercent: number }[]>([])
  const [alertOn, setAlertOn] = useState(false)
  const threshold = 70

  const sendFrame = async (blob: Blob) => {
    try {
      const form = new FormData()
      form.append('file', blob, 'frame.jpg')
      const res = await fetch(`${BACKEND_URL}/analyze`, { method: 'POST', body: form })
      if (!res.ok) return
      const data = await res.json()
      setFocusPercent(data.focusPercent)
      setEvents(data.events)
    } catch (e) {
      // ignore transient errors
    }
  }

  useWebcamCapture(1000, sendFrame)

  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/history`)
        if (!res.ok) return
        const data = await res.json()
        setHistory30(data.last30min)
        setHistory7(data.last7days)
        const latest = data.last30min.length ? data.last30min[data.last30min.length - 1].focusPercent : 100
        setAlertOn(latest < threshold)
      } catch {}
    }, 5000)
    return () => clearInterval(t)
  }, [])

  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', padding: 16, color: '#111', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ marginBottom: 8 }}>Student Focus</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ background: 'white', borderRadius: 12, padding: 16, boxShadow: '0 1px 2px rgba(0,0,0,0.06)' }}>
          <h3>Current Focus</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ width: 120, height: 120, borderRadius: '50%', background: '#e2e8f0', display: 'grid', placeItems: 'center', fontSize: 24, fontWeight: 700 }}>
              {Math.round(focusPercent)}%
            </div>
            <div>
              <div style={{ fontSize: 12, color: '#64748b' }}>Recent events</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                {events.map((e, i) => (
                  <span key={i} style={{ padding: '4px 8px', borderRadius: 999, background: '#eef2ff', color: '#3730a3', fontSize: 12 }}>{e}</span>
                ))}
                {!events.length && <span style={{ color: '#64748b', fontSize: 12 }}>None</span>}
              </div>
            </div>
          </div>
          {alertOn && (
            <div style={{ marginTop: 12, padding: 12, borderRadius: 8, background: '#fef3c7', color: '#92400e', fontWeight: 600 }}>
              Focus low. Please refocus.
            </div>
          )}
        </div>

        <div style={{ background: 'white', borderRadius: 12, padding: 16, boxShadow: '0 1px 2px rgba(0,0,0,0.06)' }}>
          <h3>Last 30 Minutes</h3>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={history30.map(p => ({ x: new Date(p.time * 1000).toLocaleTimeString(), y: p.focusPercent }))}>
              <defs>
                <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" hide/>
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Area type="monotone" dataKey="y" stroke="#2563eb" fillOpacity={1} fill="url(#g)"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: 'white', borderRadius: 12, padding: 16, boxShadow: '0 1px 2px rgba(0,0,0,0.06)' }}>
          <h3>Weekly Average</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={history7.map(p => ({ x: new Date(p.time * 1000).toLocaleDateString(), y: p.focusPercent }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="y" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: 'white', borderRadius: 12, padding: 16, boxShadow: '0 1px 2px rgba(0,0,0,0.06)' }}>
          <h3>Privacy</h3>
          <p style={{ color: '#475569', fontSize: 14 }}>Frames are processed locally and not uploaded to cloud. Audio is not recorded.</p>
        </div>
      </div>
    </div>
  )
}
