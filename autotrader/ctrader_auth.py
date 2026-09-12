"""Tokens OAuth de cTrader Open API: carga, intercambio y renovacion."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import requests

TOKEN_URL = "https://openapi.ctrader.com/apps/token"
AUTH_URL = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"


def auth_url(client_id: str, redirect_uri: str) -> str:
    return f"{AUTH_URL}?client_id={client_id}&redirect_uri={redirect_uri}&scope=trading"


def _save(path: str, data: dict) -> dict:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(data.get("expiresIn", 0)))
    data["expires_at"] = expires_at.isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(path, 0o600)
    return data


def exchange_code(code: str, client_id: str, client_secret: str, redirect_uri: str, path: str) -> dict:
    if "code=" in code:  # se pego la URL de redireccion completa
        code = code.split("code=", 1)[1].split("&", 1)[0]
    params = {"grant_type": "authorization_code", "code": code.strip(), "redirect_uri": redirect_uri,
              "client_id": client_id, "client_secret": client_secret}
    resp = requests.get(TOKEN_URL, params=params, timeout=20)
    data = resp.json()
    if resp.status_code != 200 or "accessToken" not in data:
        raise RuntimeError(f"cTrader token {resp.status_code}: {data}")
    return _save(path, data)


def refresh(refresh_token: str, client_id: str, client_secret: str, path: str) -> dict:
    params = {"grant_type": "refresh_token", "refresh_token": refresh_token,
              "client_id": client_id, "client_secret": client_secret}
    resp = requests.get(TOKEN_URL, params=params, timeout=20)
    data = resp.json()
    if resp.status_code != 200 or "accessToken" not in data:
        raise RuntimeError(f"cTrader refresh {resp.status_code}: {data}")
    return _save(path, data)


def load_access_token(path: str, client_id: str, client_secret: str, env_token: str = "", env_refresh: str = "") -> str:
    """Devuelve un access token valido. Prioridad: archivo en state/ > variables de entorno.

    Si el token del archivo caduca en menos de 3 dias, se renueva con el refresh token.
    """
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        expires_at = datetime.fromisoformat(data.get("expires_at", "1970-01-01T00:00:00+00:00"))
        if expires_at - datetime.now(timezone.utc) < timedelta(days=3) and data.get("refreshToken"):
            data = refresh(data["refreshToken"], client_id, client_secret, path)
        return data["accessToken"]
    if env_token:
        return env_token
    if env_refresh:
        return refresh(env_refresh, client_id, client_secret, path)["accessToken"]
    raise RuntimeError("No hay token de cTrader: ejecuta scripts/ctrader_token.py <code> o define CTRADER_ACCESS_TOKEN")
