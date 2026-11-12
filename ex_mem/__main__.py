"""Interactive chat with persistent memory:  python -m ex_mem"""
from ex_mem.client import ExMemClient


def main() -> None:
    client = ExMemClient()
    pruned = client.forget()
    if pruned:
        print(f"[forgot {pruned} stale memories]")
    print("ex_mem chat — /memories to list, /quit to exit")
    turn = 0
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/memories":
            for m in client.store.list_live():
                print(f"  - [{m.topic}] {m.text}")
            continue
        turn += 1
        out = client.chat(msg, source_turn=f"turn_{turn}")
        print(f"bot> {out['response']}")
        if out["signal"]["signal"] != "NONE":
            print(f"     [signal: {out['signal']['signal']} — {out['signal']['description']}]")


if __name__ == "__main__":
    main()
