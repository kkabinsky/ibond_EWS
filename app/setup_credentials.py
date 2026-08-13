# -*- coding: utf-8 -*-
"""
setup_credentials.py -- one-shot, safe setup of your ThaiBMA / iBond credentials.

WHY THIS EXISTS
    Your password must never be typed into a chat, a source file, a command line
    (it would land in the shell history) or a screenshot. This script takes it
    straight from your keyboard with hidden input and hands it to Windows, so it
    is stored only in your own user profile.

WHAT IT DOES
    1. asks for your username and password  (password input is hidden)
    2. stores them as USER environment variables via `setx`
    3. immediately tests the connection to ThaiBMA / iBond
    4. never prints, logs or copies the password anywhere

RUN
    python setup_credentials.py            # set + test
    python setup_credentials.py --test     # only test what is already stored
    python setup_credentials.py --clear    # remove the stored credentials
"""
from __future__ import annotations

import getpass
import os
import subprocess
import sys

VARS = ("THAIBMA_USER", "THAIBMA_PASS", "THAIBMA_API_KEY")


def _setx(name: str, value: str) -> bool:
    """Store a user-level environment variable. Uses a plain argument list (no shell),
    so characters like ? & ^ % in a password cannot be mangled or logged by cmd."""
    try:
        r = subprocess.run(["setx", name, value], capture_output=True, text=True, shell=False)
        return r.returncode == 0
    except Exception as ex:
        print(f"  could not store {name}: {ex}")
        return False


def test_connection() -> bool:
    try:
        import ibond_client as ib
    except Exception as ex:
        print(f"  ibond_client not importable: {ex}")
        return False
    st = ib.credentials_status()
    print("\n  credentials present:")
    print(f"    username : {'yes (' + st['user_hint'] + ')' if st['user_set'] else 'NO'}")
    print(f"    password : {'yes (hidden)' if st['pass_set'] else 'NO'}")
    print(f"    api key  : {'yes (hidden)' if st['api_key_set'] else 'not set (optional)'}")
    if not st["ready"]:
        print("\n  nothing to test yet.")
        return False
    print("\n  contacting ThaiBMA / iBond ...")
    try:
        df = ib.fetch_curve()
        print(f"  SUCCESS -- downloaded {len(df):,} rows, "
              f"{df['tau'].nunique()} tenors, "
              f"{df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
        print("\n  next:  python realtime_ews.py --live")
        print("         python yield_curve_dns.py           (uses the same session)")
        return True
    except Exception as ex:
        print("  connection did NOT succeed:\n")
        for line in str(ex).splitlines():
            print("    " + line)
        print("\n  This usually means the login worked but iBond's internal data URL")
        print("  differs from the ones guessed in ibond_client.py. To fix it:")
        print("    1. open iBond in your browser and log in")
        print("    2. go to the page with the data you want")
        print("    3. press F12 -> Network tab -> refresh the page")
        print("    4. find a request starting with 'api/', right-click -> Copy link address")
        print("    5. send that URL (URL ONLY -- never a token, cookie or password)")
        return False


def clear():
    print("removing stored credentials ...")
    for v in VARS:
        subprocess.run(["setx", v, ""], capture_output=True, text=True, shell=False)
        os.environ.pop(v, None)
    print("  done. (Windows keeps an empty variable; that is treated as 'not set'.)")


def main():
    if "--clear" in sys.argv:
        clear()
        return
    if "--test" in sys.argv:
        test_connection()
        return

    print("=" * 74)
    print("ThaiBMA / iBond credential setup")
    print("=" * 74)
    print("Your password is read with hidden input and passed straight to Windows.")
    print("It is NOT shown on screen, NOT written to any file, and NOT sent anywhere")
    print("except thaibma.or.th over HTTPS.\n")
    print("If your password was ever pasted into a chat or a document, change it on")
    print("iBond FIRST, then enter the new one here.\n")

    user = input("  iBond username: ").strip()
    if not user:
        print("  cancelled (no username).")
        return
    pw = getpass.getpass("  iBond password (hidden, nothing will appear): ")
    if not pw:
        print("  cancelled (no password).")
        return
    api = getpass.getpass("  API key if you subscribe (optional, press Enter to skip): ")

    print("\n  storing ...")
    ok = _setx("THAIBMA_USER", user) and _setx("THAIBMA_PASS", pw)
    if api:
        _setx("THAIBMA_API_KEY", api)
    # make them visible to THIS process so the test below can run right away
    os.environ["THAIBMA_USER"] = user
    os.environ["THAIBMA_PASS"] = pw
    if api:
        os.environ["THAIBMA_API_KEY"] = api
    del pw, api                                    # drop from memory promptly

    print("  stored." if ok else "  storing failed -- run this from a normal terminal.")
    print("  (other programs see them after you open a NEW terminal)")

    test_connection()


if __name__ == "__main__":
    main()
