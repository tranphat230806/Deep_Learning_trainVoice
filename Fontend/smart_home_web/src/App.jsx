import { useState } from "react";
import {
  Mic,
  DoorOpen,
  Lightbulb,
  Shield,
} from "lucide-react";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [verified, setVerified] = useState(false);
  const [owner, setOwner] = useState("Unknown");
  const [confidence, setConfidence] = useState("0%");
  const [door, setDoor] = useState("Locked");
  const [light, setLight] = useState("OFF");
  const [status, setStatus] = useState("Security mode enabled");

  // ===============================
  // VERIFY GIỌNG NÓI THẬT
  // ===============================
  const verifyVoice = async () => {
    try {
      setLoading(true);
      setStatus("Listening to microphone...");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const recorder = new MediaRecorder(stream);
      let chunks = [];

      recorder.ondataavailable = (e) => {
        chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, {
          type: "audio/webm",
        });

        const formData = new FormData();
        formData.append("audio", blob, "voice.webm");

        setStatus("Verifying voice...");

        const res = await fetch(
          "http://localhost:5000/verify",
          {
            method: "POST",
            body: formData,
          }
        );

        const data = await res.json();

        if (data.verified) {
          setVerified(true);
          setOwner("Phát");
          setConfidence(
            `${(data.score * 100).toFixed(1)}%`
          );
          setStatus("Access Granted");
        } else {
          setVerified(false);
          setOwner("Unknown");
          setConfidence(
            `${(data.score * 100).toFixed(1)}%`
          );
          setStatus("Access Denied");
        }

        setLoading(false);
      };

      recorder.start();

      setTimeout(() => {
        recorder.stop();
        stream.getTracks().forEach((t) => t.stop());
      }, 4000);
    } catch (err) {
      setStatus("Microphone error");
      setLoading(false);
    }
  };

  // ===============================
  const openDoor = () => {
    if (!verified) {
      setStatus("Access denied");
      return;
    }

    setDoor("Opened");
    setStatus("Main door opened");
  };

  const toggleLight = () => {
    if (!verified) {
      setStatus("Access denied");
      return;
    }

    setLight(light === "ON" ? "OFF" : "ON");
  };

  const lockdown = () => {
    setVerified(false);
    setOwner("Unknown");
    setDoor("Locked");
    setStatus("Security Lockdown Enabled");
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-5xl font-bold tracking-wide text-cyan-400">
          JARVIS HOME AI
        </h1>

        <p className="text-gray-400 mt-2">
          Stark-level Smart Home Security
        </p>
      </div>

      {/* Status */}
      <div className="bg-gray-900 rounded-2xl p-6 shadow-lg mb-6">
        <h2 className="text-xl font-semibold mb-4">
          System Status
        </h2>

        <p className="text-green-400">{status}</p>
      </div>

      {/* Voice Auth */}
      <div className="bg-gray-900 rounded-2xl p-6 shadow-lg mb-6 text-center">
        <Mic
          className={`mx-auto mb-3 ${
            loading
              ? "text-red-400 animate-pulse"
              : "text-cyan-400"
          }`}
          size={40}
        />

        <h2 className="text-xl font-semibold mb-2">
          Voice Authentication
        </h2>

        <button
          onClick={verifyVoice}
          disabled={loading}
          className="mt-3 px-6 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-full transition"
        >
          {loading ? "Listening..." : "Tap to Verify"}
        </button>

        <div className="mt-6 text-left">
          <h3 className="font-semibold mb-2">
            Identity Result
          </h3>

          <p>
            Owner:{" "}
            <span
              className={
                verified
                  ? "text-green-400"
                  : "text-red-400"
              }
            >
              {owner}
            </span>
          </p>

          <p>
            Confidence:{" "}
            <span className="text-yellow-400">
              {confidence}
            </span>
          </p>

          <p>
            Access:{" "}
            <span
              className={
                verified
                  ? "text-green-500 font-bold"
                  : "text-red-500 font-bold"
              }
            >
              {verified ? "GRANTED" : "DENIED"}
            </span>
          </p>
        </div>
      </div>

      {/* Home Core */}
      <div className="bg-gray-900 rounded-2xl p-6 shadow-lg mb-6">
        <h2 className="text-xl font-semibold mb-4">
          Home Core
        </h2>

        <div className="grid grid-cols-3 gap-4">
          <button
            onClick={openDoor}
            className="flex flex-col items-center p-4 bg-gray-800 rounded-xl hover:bg-gray-700"
          >
            <DoorOpen />
            <span className="mt-2">Main Door</span>
          </button>

          <button
            onClick={toggleLight}
            className="flex flex-col items-center p-4 bg-gray-800 rounded-xl hover:bg-gray-700"
          >
            <Lightbulb />
            <span className="mt-2">Lights</span>
          </button>

          <button
            onClick={lockdown}
            className="flex flex-col items-center p-4 bg-red-600 hover:bg-red-700 rounded-xl"
          >
            <Shield />
            <span className="mt-2">Lockdown</span>
          </button>
        </div>
      </div>

      {/* Live Status */}
      <div className="bg-gray-900 rounded-2xl p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-4">
          Live Status
        </h2>

        <ul className="space-y-2 text-gray-300">
          <li>
            Owner:{" "}
            <span
              className={
                verified
                  ? "text-green-400"
                  : "text-red-400"
              }
            >
              {verified ? "Online" : "Offline"}
            </span>
          </li>

          <li>
            Main Door:{" "}
            <span className="text-green-400">
              {door}
            </span>
          </li>

          <li>
            Lights:{" "}
            <span className="text-yellow-400">
              {light}
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
}