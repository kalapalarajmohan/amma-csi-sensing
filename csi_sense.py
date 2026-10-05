#!/usr/bin/env python3
"""csi_sense.py - one-file reader for ESP32 WiFi CSI sent over UDP (port 5005).
Prints movement and breathing rate on demand. Same signal code as amma_40nights.py.
Needs: python3, numpy, scipy.   Run:  python3 csi_sense.py
Options: --packets N (window size, default 900, about 30 s)  --watch  --json  --port P
Packet format expected: first 104 bytes = 52 subcarriers as int8 (imag, real) pairs."""
import argparse, json, socket, struct, sys, time
import numpy as np
from scipy.signal import butter, filtfilt
try:
    from scipy.signal import find_peaks
except ImportError:  # scipy older than 1.1 (2018): small stand-in with the same behaviour
    def find_peaks(x, distance=None):
        x = np.asarray(x)
        idx = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]
        if distance and len(idx) > 1:
            keep = []
            for i in sorted(idx, key=lambda k: -x[k]):
                if all(abs(i - j) >= distance for j in keep):
                    keep.append(i)
            idx = sorted(keep)
        return np.array(idx, dtype=int), {}

N_SC = 52

def safe(v, default=0.0):
    try:
        f = float(v)
        return default if (f != f or abs(f) == float("inf")) else f
    except Exception:
        return default

def collect(port, n_packets, timeout=15):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    sock.bind(("0.0.0.0", port))
    amps, times = [], []
    try:
        while len(amps) < n_packets:
            data, _ = sock.recvfrom(65535)
            if len(data) < 2 * N_SC:
                continue
            raw = struct.unpack("%db" % (2 * N_SC), data[:2 * N_SC])
            imag = np.array(raw[0::2], dtype=np.float32)
            real = np.array(raw[1::2], dtype=np.float32)
            amps.append(np.abs(real + 1j * imag))
            times.append(time.time())
    except socket.timeout:
        print("No packets on UDP port %d for %d s. Check the ESP32 is powered, on the "
              "same WiFi, and sending to this Jetson's IP." % (port, timeout), file=sys.stderr)
        return None, None
    finally:
        sock.close()
    return np.stack(amps), np.array(times)

def extract(amp, times):
    g = safe(amp.mean(), 1.0)
    energy = amp.mean(axis=1)
    movement = safe(np.std(energy))
    breathing, fs = None, None
    try:
        fs = (len(times) - 1) / (times[-1] - times[0])
        if fs >= 4:
            nyq = fs / 2
            b, a = butter(4, [0.15 / nyq, 0.5 / nyq], btype="band")
            filt = filtfilt(b, a, energy)
            pks, _ = find_peaks(filt, distance=int(fs * 2))
            if len(pks) >= 2:
                rate = 60.0 / np.mean(np.diff(pks) / fs)
                breathing = round(rate, 1) if 6 <= rate <= 30 else None
    except Exception:
        pass
    activity = "active" if movement > 0.8 else "restless" if movement > 0.3 else "still"
    return {"movement": round(movement, 3), "activity": activity,
            "breathing_bpm": breathing, "packets": len(times),
            "seconds": round(float(times[-1] - times[0]), 1),
            "pkts_per_s": round(fs, 1) if fs else None}

def main():
    p = argparse.ArgumentParser(description="Movement and breathing from ESP32 CSI")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--packets", type=int, default=900)
    p.add_argument("--watch", action="store_true", help="repeat forever")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args()
    while True:
        print("Listening on UDP %d for %d packets..." % (args.port, args.packets), file=sys.stderr)
        amp, times = collect(args.port, args.packets)
        if amp is None:
            return 1
        r = extract(amp, times)
        if args.json:
            print(json.dumps(r))
        else:
            b = "%.1f breaths/min" % r["breathing_bpm"] if r["breathing_bpm"] else "not detected"
            print("movement %.3f (%s) | breathing %s | %d packets in %.1f s (%s/s)"
                  % (r["movement"], r["activity"], b, r["packets"], r["seconds"], r["pkts_per_s"]))
        if not args.watch:
            return 0

if __name__ == "__main__":
    sys.exit(main())
