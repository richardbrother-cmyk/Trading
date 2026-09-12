"""Intercambia un codigo de autorizacion de cTrader Open API por tokens y los guarda en state/.

Uso:
  python scripts/ctrader_token.py <code>          # primer intercambio
  python scripts/ctrader_token.py --refresh       # renovar con el refresh token guardado
  python scripts/ctrader_token.py --url           # imprimir el enlace de autorizacion
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://openapi.ctrader.com/apps/token"
AUTH_URL = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"


def main() -> int:
    load_dotenv(".env")
    client_id = os.environ["CTRADER_CLIENT_ID"]
    client_secret = os.environ["CTRADER_CLIENT_SECRET"]
    redirect_uri = os.getenv("CTRADER_REDIRECT_URI", "https://localhost")
    state_dir = os.getenv("STATE_DIR", "state")
    path = os.path.join(state_dir, "ctrader_tokens.json")
    os.makedirs(state_dir, exist_ok=True)

    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    arg = sys.argv[1]
    if arg == "--url":
        print(f"{AUTH_URL}?client_id={client_id}&redirect_uri={redirect_uri}&scope=trading")
        return 0
    if arg == "--refresh":
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        params = {"grant_type": "refresh_token", "refresh_token": saved["refreshToken"],
                  "client_id": client_id, "client_secret": client_secret}
    else:
        code = arg.strip()
        if "code=" in code:  # se pego la URL completa
            code = code.split("code=", 1)[1].split("&", 1)[0]
        params = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
                  "client_id": client_id, "client_secret": client_secret}

    resp = requests.get(TOKEN_URL, params=params, timeout=20)
    data = resp.json()
    if resp.status_code != 200 or "accessToken" not in data:
        print(f"Error {resp.status_code}: {data}", file=sys.stderr)
        return 1
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(data.get("expiresIn", 0)))
    data["expires_at"] = expires_at.isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(path, 0o600)
    print(f"Tokens guardados en {path}; el access token caduca el {expires_at.date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
