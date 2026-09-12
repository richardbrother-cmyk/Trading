"""Autorizacion OAuth de cTrader Open API.

Uso:
  python scripts/ctrader_token.py --url            # imprimir el enlace de autorizacion
  python scripts/ctrader_token.py <code o URL>     # canjear el codigo por tokens
  python scripts/ctrader_token.py --refresh        # renovar con el refresh token guardado
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ctrader_auth  # noqa: E402


def main() -> int:
    load_dotenv(".env")
    client_id = os.environ["CTRADER_CLIENT_ID"]
    client_secret = os.environ["CTRADER_CLIENT_SECRET"]
    redirect_uri = os.getenv("CTRADER_REDIRECT_URI", "https://localhost")
    path = os.path.join(os.getenv("STATE_DIR", "state"), "ctrader_tokens.json")
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    arg = sys.argv[1]
    if arg == "--url":
        print(ctrader_auth.auth_url(client_id, redirect_uri))
        return 0
    if arg == "--refresh":
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        data = ctrader_auth.refresh(saved["refreshToken"], client_id, client_secret, path)
    else:
        data = ctrader_auth.exchange_code(arg, client_id, client_secret, redirect_uri, path)
    print(f"Tokens guardados en {path}; el access token caduca el {data['expires_at'][:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
