"""lr4070.py: localread.py pinned to gpu 0 (4070 Ti), HTTP timeout 1800 s, answer cap via LR_TWEET, max_tokens via LR_MAXTOK."""
import os, sys
sys.path.insert(0, r"C:\Users\john\Documents\Tools\llm")
import localread as L
L.TWEET = int(os.environ.get("LR_TWEET", "300"))
_http = L.http
L.http = lambda port, path, body=None, timeout=600: _http(port, path, body, 1800 if timeout >= 600 else timeout)
_free = L.free_vram_mb
L.free_vram_mb = lambda index=0: (_free(0) if index == 0 else None)   # gpu 1 reported unusable -> picker takes gpu 0
_tweet = L.tweet
L.tweet = lambda text, max_chars=None: _tweet(text, max_chars or L.TWEET)
_ask = L.ask
L.ask = lambda port, system, user, max_tokens: _ask(port, system, user, int(os.environ.get("LR_MAXTOK", max_tokens)))
if __name__ == "__main__":
    L.main()
