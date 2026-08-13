# -*- coding: utf-8 -*-
"""
ibond_grpc.py -- gRPC-Web client for ThaiBMA iBond.

iBond is NOT a REST service. Its front end talks gRPC-Web, which is why plain
POSTs to /api/... answer 405. The service and message definitions below were read
out of iBond's own public JavaScript bundle (/static/js/main.*.js):

  AUTH   POST {base}/grpc/authen-grpc/authen.AuthenGrpcService/AuthenticateV2
         request  authen.AuthenticateMessage   1=userName(str)  2=password(str)
         reply    authen.AuthenticateReply     5=token  6=refreshToken  7=expiredDate

  CURVE  POST {base}/grpc/yieldcurve-grpc/yieldcurve.YieldCurveGrpcService/GetYieldCurveByAsOf
         request  yieldcurve.YieldCurveRequest 1=asof(Timestamp) 2=data(str)
         reply    IEnumerable_YieldCurveReply  1=repeated YieldCurveReply
                  YieldCurveReply              1=asof(Timestamp) 2=ttm(Double)
                                               3=yield(Double)   4=duration(Double)

  DATES  POST {base}/grpc/yieldcurve-grpc/yieldcurve.YieldCurveGrpcService/GetYieldCurveAvaliableDate

Only the handful of protobuf constructs these messages need are implemented here,
so there is no protoc / grpcio dependency.

SECURITY: this module never stores a password. It reads THAIBMA_USER and
THAIBMA_PASS from your own environment (set them with setup_credentials.py) and
sends them only to thaibma.or.th over HTTPS.
"""
from __future__ import annotations

import base64
import os
import struct
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = os.environ.get("IBOND_BASE", "https://www.ibond.thaibma.or.th")
AUTH_SVC = "/grpc/authen-grpc/authen.AuthenGrpcService"
CURVE_SVC = "/grpc/yieldcurve-grpc/yieldcurve.YieldCurveGrpcService"
# Verified against the live server with a probe account: "Authenticate" parses the
# username/password correctly (a bad user gets a clean grpc-status 16 "User name not
# found"), whereas "AuthenticateV2" needs an extra field and returns status 13.
AUTH_METHODS = ("Authenticate",)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TIMEOUT = 15


# ------------------------------------------------------------ protobuf -------
def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def pb_string(field: int, s: str) -> bytes:
    b = s.encode("utf-8")
    return _tag(field, 2) + _varint(len(b)) + b


def pb_int32(field: int, v: int) -> bytes:
    return _tag(field, 0) + _varint(int(v))


def pb_message(field: int, body: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(body)) + body


def pb_timestamp(field: int, dt: datetime) -> bytes:
    """google.protobuf.Timestamp = 1:seconds(int64) 2:nanos(int32)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = int(dt.timestamp())
    return pb_message(field, pb_int32(1, secs))


def pb_parse(buf: bytes) -> dict:
    """Decode a flat protobuf message -> {field_no: [raw values]}."""
    out, i, n = {}, 0, len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _read_varint(buf, i)
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wire == 1:
            v, i = buf[i:i + 8], i + 8
        elif wire == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            break
        out.setdefault(field, []).append(v)
    return out


def _read_varint(buf: bytes, i: int):
    shift, res = 0, 0
    while i < len(buf):
        b = buf[i]; i += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, i
        shift += 7
    return res, i


def _double_value(raw: bytes):
    """google.protobuf.DoubleValue -> float (field 1, wire 1 = fixed64)."""
    f = pb_parse(raw)
    if 1 in f and isinstance(f[1][0], (bytes, bytearray)) and len(f[1][0]) == 8:
        return struct.unpack("<d", f[1][0])[0]
    return None


def _timestamp_value(raw: bytes):
    f = pb_parse(raw)
    if 1 in f and isinstance(f[1][0], int):
        return datetime.fromtimestamp(f[1][0], tz=timezone.utc)
    return None


# ---------------------------------------------------------- gRPC-Web ---------
def grpc_frame(payload: bytes) -> bytes:
    """gRPC-Web data frame: flag byte 0x00 + 4-byte big-endian length + payload."""
    return b"\x00" + struct.pack(">I", len(payload)) + payload


def b64_stream_decode(data: bytes) -> bytes:
    """Decode a grpc-web-TEXT body.

    The server does not send one big base64 blob: it emits a base64 chunk per gRPC
    frame (message, then trailers), and those chunks are concatenated on the wire.
    Each chunk carries its own '=' padding, so decoding the concatenation in one go
    raises "Incorrect padding" as soon as a response is large enough to be split
    (e.g. a real login reply carrying a JWT). Decode segment by segment instead.
    """
    s = bytes(data).translate(None, b"\r\n \t")
    if not s:
        return b""
    try:                                     # fast path: single, well-formed chunk
        return base64.b64decode(s + b"=" * (-len(s) % 4))
    except Exception:
        pass
    out, i, n = b"", 0, len(s)
    while i < n:
        j = s.find(b"=", i)
        if j < 0:
            seg, i = s[i:], n
        else:
            k = j
            while k < n and s[k:k + 1] == b"=":
                k += 1
            seg, i = s[i:k], k
        if not seg:
            break
        try:
            out += base64.b64decode(seg + b"=" * (-len(seg) % 4))
        except Exception:
            continue                          # skip an unusable fragment, keep going
    if not out:
        raise ValueError("could not base64-decode the grpc-web-text response")
    return out


def grpc_unframe(body: bytes):
    """Yield (flag, payload) for each frame. flag 0x80 marks the trailer frame."""
    i = 0
    while i + 5 <= len(body):
        flag = body[i]
        ln = struct.unpack(">I", body[i + 1:i + 5])[0]
        payload = body[i + 5:i + 5 + ln]
        yield flag, payload
        i += 5 + ln


class IBondGrpc:
    """iBond's own front end builds its gRPC client with format="text" (seen in the
    JS bundle), i.e. grpc-web-TEXT: each frame is base64-encoded and the content type
    is application/grpc-web-text. Sending raw binary makes the .NET server deserialize
    an empty message -> NullReferenceException (grpc-status 13). So we default to text."""

    def __init__(self, base: str = BASE, text_mode: bool = True):
        self.base = base.rstrip("/")
        self.token = None
        self.text_mode = text_mode
        self.s = requests.Session()
        ctype = "application/grpc-web-text" if text_mode else "application/grpc-web+proto"
        self.s.headers.update({
            "User-Agent": UA, "Origin": self.base, "Referer": self.base + "/",
            "Content-Type": ctype, "Accept": ctype,
            "X-Grpc-Web": "1", "X-User-Agent": "grpc-web-javascript/0.1",
        })

    # -- transport ---------------------------------------------------------
    def call(self, service: str, method: str, body: bytes, timeout: int | None = None) -> bytes:
        """`timeout` overrides the module default -- bulk queries such as the full
        corporate-bond search return megabytes and need much longer than a lookup."""
        url = f"{self.base}{service}/{method}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        frame = grpc_frame(body)
        payload = base64.b64encode(frame) if self.text_mode else frame
        r = self.s.post(url, data=payload, headers=headers,
                        timeout=timeout or TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"{method}: HTTP {r.status_code} "
                               f"({r.text[:160] if r.text else 'no body'})")
        # header-level grpc-status (some servers put it here)
        hstatus = r.headers.get("grpc-status")
        if hstatus not in (None, "0"):
            raise RuntimeError(f"{method}: grpc-status {hstatus} "
                               f"{r.headers.get('grpc-message', '')}")
        raw = b64_stream_decode(r.content) if self.text_mode else r.content
        data = b""
        for flag, part in grpc_unframe(raw):
            if flag == 0:
                data += part
            elif flag & 0x80:                        # trailer frame
                txt = part.decode("utf-8", "ignore")
                st = dict(kv.split(":", 1) for kv in txt.replace("\r", "").split("\n")
                          if ":" in kv)
                code = (st.get("grpc-status") or "0").strip()
                if code not in ("", "0"):
                    raise RuntimeError(f"{method}: grpc-status {code} "
                                       f"{st.get('grpc-message', '').strip()}")
        return data

    # -- auth --------------------------------------------------------------
    def login(self, user: str | None = None, password: str | None = None) -> dict:
        if not (user and password):
            try:                                   # also reads the persisted user env,
                import ibond_client as ic          # so a fresh `setx` works right away
                u, p, _k = ic._creds()
            except Exception:
                u = p = None
            user = user or u or os.environ.get("THAIBMA_USER")
            password = password or p or os.environ.get("THAIBMA_PASS")
        if not (user and password):
            raise RuntimeError("THAIBMA_USER / THAIBMA_PASS not set — run setup_credentials.py")
        body = pb_string(1, user) + pb_string(2, password)
        errors = []
        # iBond writes a session row on login, so rapid repeated logins can hit a
        # server-side SQL deadlock. Retry those (and only those) with a backoff.
        for attempt in range(3):
            try:
                return self._login_once(body)
            except RuntimeError as ex:
                msg = str(ex)
                transient = ("deadlock" in msg.lower() or "timeout" in msg.lower()
                             or "grpc-status 14" in msg)
                if not transient or attempt == 2:
                    raise
                errors.append(msg)
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("login failed — " + " | ".join(errors))

    def _login_once(self, body: bytes) -> dict:
        errors = []
        for method in AUTH_METHODS:
            try:
                reply = self.call(AUTH_SVC, method, body)
                f = pb_parse(reply)

                def s(n):
                    v = f.get(n, [b""])[0]
                    return v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else ""
                # AuthenticateReply puts the JWT in field 5; fall back to whichever
                # string field actually looks like a token if the shape ever changes.
                self.token = s(5) or s(3)
                if not self.token:
                    cand = [v.decode("utf-8", "ignore")
                            for vals in f.values() for v in vals
                            if isinstance(v, (bytes, bytearray)) and len(v) > 40]
                    self.token = next((t for t in cand if t.count(".") >= 2), "")
                if self.token:
                    return {"user_id": s(1), "first_name": s(2), "user_name": s(4),
                            "expires": s(7), "token_len": len(self.token), "auth_method": method}
                errors.append(f"{method}: server replied but no token field was present")
            except Exception as ex:
                msg = str(ex)
                if "grpc-status 16" in msg:            # UNAUTHENTICATED
                    raise RuntimeError(
                        "iBond rejected the credentials (grpc-status 16: "
                        + msg.split("grpc-status 16")[-1].strip()
                        + ").\nRe-run:  python setup_credentials.py   and enter the "
                        "current iBond username/password.") from None
                errors.append(f"{method}: {msg}")
        raise RuntimeError("authentication returned no token — " + " | ".join(errors))

    # -- data --------------------------------------------------------------
    def available_dates(self) -> list:
        try:
            reply = self.call(CURVE_SVC, "GetYieldCurveAvaliableDate", b"")
        except Exception:
            return []
        out = []
        for _fno, vals in pb_parse(reply).items():
            for v in vals:
                if isinstance(v, (bytes, bytearray)):
                    ts = _timestamp_value(v)
                    if ts:
                        out.append(ts.date())
                    else:
                        txt = v.decode("utf-8", "ignore").strip()
                        if len(txt) >= 8:
                            out.append(txt)
        return out

    def yield_curve(self, asof: datetime, data: str = "") -> pd.DataFrame:
        """GetYieldCurveByAsOf -> tidy [date, tau, yield, duration]."""
        body = pb_timestamp(1, asof) + (pb_string(2, data) if data else b"")
        reply = self.call(CURVE_SVC, "GetYieldCurveByAsOf", body)
        rows = []
        for item in pb_parse(reply).get(1, []):          # repeated YieldCurveReply
            f = pb_parse(item)
            d = _timestamp_value(f[1][0]) if 1 in f else None
            ttm = _double_value(f[2][0]) if 2 in f else None
            yld = _double_value(f[3][0]) if 3 in f else None
            dur = _double_value(f[4][0]) if 4 in f else None
            if ttm is None or yld is None:
                continue
            rows.append({"date": (d.date() if d else asof.date()),
                         "tau": ttm, "yield": yld, "duration": dur})
        return pd.DataFrame(rows)

    def bid_yield_table(self, year: int) -> str:
        """GetBidYieldDataTable(year) -> the raw table string iBond renders."""
        reply = self.call(CURVE_SVC, "GetBidYieldDataTable", pb_int32(1, year))
        f = pb_parse(reply)
        return f[1][0].decode("utf-8", "ignore") if 1 in f else ""


# ------------------------------------------------------------------ api ------
def fetch_curve_history(dates) -> pd.DataFrame:
    """Log in once and pull the curve for each date in `dates`."""
    c = IBondGrpc()
    who = c.login()
    print(f"  logged in as {who.get('user_name') or who.get('user_id')} "
          f"(token {who['token_len']} chars)")
    frames = []
    for d in dates:
        dt = pd.Timestamp(d).to_pydatetime()
        try:
            df = c.yield_curve(dt)
            if not df.empty:
                frames.append(df)
        except Exception as ex:
            print(f"    {pd.Timestamp(d):%Y-%m-%d}: {str(ex)[:90]}")
    if not frames:
        raise RuntimeError("no curve data returned")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["source"] = "ibond-grpc"
    return out.sort_values(["date", "tau"]).reset_index(drop=True)


def month_ends(start: str, end: str):
    return list(pd.date_range(start=start, end=end, freq="ME"))


if __name__ == "__main__":
    import sys
    start = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else "2024-01"
    end = sys.argv[sys.argv.index("--end") + 1] if "--end" in sys.argv else "2025-12"
    print("iBond gRPC-Web client")
    c = IBondGrpc()
    info = c.login()
    print("  login OK:", {k: v for k, v in info.items() if k != "token_len"})
    dates = c.available_dates()
    print(f"  available dates reported: {len(dates)}"
          + (f"  (e.g. {dates[:3]})" if dates else ""))
    df = fetch_curve_history(month_ends(start, end))
    print(f"\n  downloaded {len(df):,} rows | {df['tau'].nunique()} tenors | "
          f"{df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
    print(df.head(8).to_string(index=False))
