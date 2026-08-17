"""Miner-side bet posting.  THE MODEL: miners post their own bets, by
hand, on chain -- the owner can only resolve them later.

Every FLOW trade the money layer will ever touch exists because the
miner personally put it there: `postTrade(hotkey, tradeKey, collateral)`
reserves the bet's worst case (kelly f x collateral) behind the trade
key.  The owner cannot create, resize, or extend a bet -- its entire
write surface is resolving posted bets (losses bounded by each bet),
settling the weekly pool (zero-sum enforced), and repairing rounding
dust.  What is never posted can never lose -- or win -- a single rao.

WHO SIGNS.  The native path is the HOTKEY: positions are booked to the
raw 32-byte hotkey, which IS an sr25519 public key, so the miner signs
each bet with the same key they mine with and the contract verifies it
through the chain's sr25519 precompile (`postTradeSigned`).  The EVM
account that relays the transaction only pays gas -- the signed digest
binds chain, contract, hotkey, trade key, size and a strictly
increasing nonce, so a relayer can neither alter a bet nor replay one.
The depositor EVM key (the coldkey's EVM face that funded the
position) can also post directly (`postTrade`) as a fallback.

A bet is IMMUTABLE once posted: no resize, no cancel -- direction
and levels are not on chain (they sit in the encrypted R2 payload
under the public timelock), so the poster already knows them while
everyone else waits; a changeable bet would be a free option.  Its reserve expires on-chain
by the regime's horizon ceiling plus 48h (derived from the trade key),
and `sweepExpired` is permissionless, so owner silence can never lock
margin.  Post the bet BEFORE (or as) you upload the payload that
carries the matching trade id: the settlement daemon only credits
bets that were on chain when the trade opened -- a bet posted after
the fact is voided both ways, because post-hoc posting would otherwise
be a free look at the tape.

Opening the position is hotkey-authorized too (`addCollateralSigned`):
your hotkey signs the funder and refund coldkey, so nobody can squat
your slot by funding 1 alpha first.  `fund` does that; `evict`
force-exits a flat book back to the recorded refund coldkey and frees
the slot.  Wire bets into your submission loop so they post as trades
open; from the CLI (--wallet signs with your local bittensor hotkey;
FLOW_EVM_KEY pays gas and, on `fund`, is the funder):

    export FLOW_RPC=https://lite.chain.opentensor.ai
    export FLOW_CONTRACT=0xD9c805202b16671A2901307fBC9A8750E2453427
    export FLOW_EVM_KEY=0x...           # funder; pays gas
    python3 flow_post.py fund --hotkey 5F... --alpha 10 \
        --wallet mywallet --wallet-hotkey myhotkey     # open (or top up)
    python3 flow_post.py book --hotkey 5F...
    python3 flow_post.py post --hotkey 5F... --vector-json vec.json \
        --wallet mywallet --wallet-hotkey myhotkey     # all live intents
    python3 flow_post.py post --hotkey 5F... --regime C --trade-id 7 \
        --f 0.05 --wallet mywallet                     # one bet
    python3 flow_post.py status --hotkey 5F... --keys C:7 D:2
    python3 flow_post.py audit  --hotkey 5F...
    python3 flow_post.py evict --hotkey 5F... \
        --wallet mywallet --wallet-hotkey myhotkey     # flat only

`audit` replays the contract's own event log and proves that every
debit stayed within the bet you posted -- trusting nothing the owner
runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from flow import DIM, FIELDS_PER_REGIME, N_REGIMES
from flow_collateral import (RAO_PER_ALPHA, CollateralClient, ss58_to_bytes32,
                           trade_key)

REGIME_NAMES = "ABCD"


# ------------------------------------------------------------ composition

def bets_for_vector(vector: List[float], collateral_alpha: float
                    ) -> Dict[Tuple[int, int], float]:
    """The bets implied by a dim-28 payload vector: every regime slot
    carrying an intent (direction != 0, trade_id > 0, f > 0) maps
    (regime, trade_id) -> bet_alpha = f x collateral.  Deterministic
    and side-effect free."""
    if len(vector) != DIM:
        raise ValueError(f"vector must be {DIM} floats")
    out: Dict[Tuple[int, int], float] = {}
    for r in range(N_REGIMES):
        seg = vector[r * FIELDS_PER_REGIME:(r + 1) * FIELDS_PER_REGIME]
        d, f, tid = seg[0], seg[1], int(round(seg[6]))
        if d == 0 or tid <= 0 or f <= 0:
            continue
        out[(r, tid)] = float(f) * float(collateral_alpha)
    return out


# --------------------------------------------------- hotkey signing

def load_hotkey_keypair(wallet_name: str, hotkey_name: str = "default",
                        wallet_path: Optional[str] = None):
    """The miner's own hotkey keypair from the local bittensor wallet
    (the same key the mining box already runs).  Signing happens on
    this machine; the seed never travels."""
    from bittensor_wallet import Wallet
    kwargs = dict(name=wallet_name, hotkey=hotkey_name)
    if wallet_path:
        kwargs["path"] = wallet_path
    return Wallet(**kwargs).get_hotkey()


def sign_post(client: CollateralClient, hotkey: str, key: int,
              collateral_alpha: float, keypair,
              nonce: Optional[int] = None) -> dict:
    """Sign one bet with the hotkey.  The digest binds chain, contract,
    hotkey, trade key, size and a strictly increasing nonce (defaults
    to max(chain nonce + 1, unix time)); the result can be relayed by
    ANY account without gaining authority over it."""
    if bytes(keypair.public_key) != ss58_to_bytes32(hotkey):
        raise ValueError("wallet hotkey does not match the position hotkey")
    if nonce is None:
        nonce = max(client.post_nonce(hotkey) + 1, int(time.time()))
    collateral_rao = int(round(collateral_alpha * RAO_PER_ALPHA))
    digest = client.post_digest(hotkey, key, collateral_rao, nonce)
    sig = bytes(keypair.sign(digest))
    return {"hotkey": hotkey, "trade_key": int(key),
            "collateral_alpha": collateral_alpha, "nonce": int(nonce),
            "signature": "0x" + sig.hex()}


# --------------------------------------------------------------- actions

def post_bet(client: CollateralClient, hotkey: str, regime: int, tid: int,
             collateral_alpha: float, keypair=None) -> dict:
    """Post one bet, idempotently: a key that already carries a live
    bet is reported, not re-posted.  With `keypair` (the miner's
    hotkey) the bet is hotkey-signed and merely relayed by the
    client's EVM account; without it the client's EVM key must be the
    position's depositor."""
    key = trade_key(regime, tid)
    rec = {"regime": REGIME_NAMES[regime], "trade_id": int(tid),
           "trade_key": key, "collateral_alpha": round(collateral_alpha, 9)}
    info = client.trade_info(hotkey, key)
    if info["expiry"] != 0:
        rec.update(status="already-posted",
                   collateral_alpha=info["exposure_rao"] / RAO_PER_ALPHA)
        return rec
    if keypair is not None:
        env = sign_post(client, hotkey, key, collateral_alpha, keypair)
        tx = client.post_trade_signed(hotkey, key, collateral_alpha,
                                      env["nonce"],
                                      bytes.fromhex(env["signature"][2:]))
        rec.update(status="posted", via="hotkey-signature",
                   nonce=env["nonce"], tx=tx)
    else:
        tx = client.post_trade(hotkey, key, collateral_alpha)
        rec.update(status="posted", via="depositor", tx=tx)
    return rec


def post_for_vector(client: CollateralClient, hotkey: str,
                    vector: List[float], keypair=None,
                    collateral_alpha: Optional[float] = None) -> List[dict]:
    """Post every live intent in a payload vector that is not already
    on chain, sized f x current collateral.  Do this BEFORE (or as)
    the payload uploads: only bets on chain when the trade opens carry
    money."""
    if collateral_alpha is None:
        collateral_alpha = client.active_collateral_of(hotkey)
    results = []
    for (r, tid), bet in sorted(bets_for_vector(vector, collateral_alpha).items()):
        try:
            results.append(post_bet(client, hotkey, r, tid, bet, keypair))
        except Exception as e:  # noqa: BLE001 - per-bet isolation
            results.append({"regime": REGIME_NAMES[r], "trade_id": tid,
                            "status": "error",
                            "error": f"{type(e).__name__}: {e}"})
    return results


# ---------------------------------------------------------------- verify

def status(client: CollateralClient, hotkey: str,
           keys: List[Tuple[int, int]]) -> List[dict]:
    """Current on-chain state for a set of (regime, trade_id) bets."""
    out = []
    for r, tid in keys:
        key = trade_key(r, tid)
        info = client.trade_info(hotkey, key)
        out.append({
            "regime": REGIME_NAMES[r], "trade_id": tid, "trade_key": key,
            "posted": info["expiry"] != 0,
            "collateral_alpha": info["exposure_rao"] / RAO_PER_ALPHA,
            "expiry": info["expiry"],
        })
    return out


def book(client: CollateralClient, hotkey: str) -> dict:
    """The position as the chain sees it right now."""
    info = client.position_info(hotkey)
    dep = info["depositor"]
    return {
        "hotkey": hotkey,
        "open": int(dep, 16) != 0,
        "depositor": dep,
        "refund_coldkey": info["refund_coldkey"],
        "balance_alpha": info["balance_rao"] / RAO_PER_ALPHA,
        "reserved_alpha": info["open_exposure_rao"] / RAO_PER_ALPHA,
        "free_alpha": max(info["balance_rao"]
                          - info["open_exposure_rao"], 0) / RAO_PER_ALPHA,
        "open_bets": info["open_count"],
        "pooled_out_alpha": info["pooled_out_rao"] / RAO_PER_ALPHA,
        "flat": (info["open_count"] == 0 and info["pooled_out_rao"] == 0),
    }


def fund_position(client: CollateralClient, hotkey: str,
                  amount_alpha: float, keypair=None,
                  refund_coldkey: Optional[str] = None) -> dict:
    """Open the slot (hotkey-signed) or top it up (depositor-only).

    Opening requires `keypair` — the hotkey signs the funder and
    refund coldkey, so a stranger funding 1 alpha cannot squat the
    slot.  A later top-up from the same depositor needs no signature.
    """
    before = client.position_info(hotkey)
    opening = int(before["depositor"], 16) == 0
    if opening and keypair is None:
        raise SystemExit(
            "opening a position needs --wallet (the hotkey must sign "
            "the funder and refund coldkey; otherwise anyone could "
            "squat the slot)")
    tx = client.add_collateral(hotkey, amount_alpha,
                               refund_coldkey=refund_coldkey,
                               keypair=keypair)
    after = book(client, hotkey)
    after.update(status="opened" if opening else "topped-up",
                 alpha=float(amount_alpha), tx=tx)
    return after


def evict_position(client: CollateralClient, hotkey: str,
                   keypair) -> dict:
    """Hotkey-signed force-exit of a FLAT position.

    Pays the balance only to the recorded refund coldkey (cannot
    redirect a rao) and frees the slot.  Anyone may relay.
    """
    if keypair is None:
        raise SystemExit("evict needs --wallet (hotkey signature)")
    info = client.position_info(hotkey)
    if int(info["depositor"], 16) == 0:
        raise SystemExit("no position to evict")
    if info["open_count"] or info["pooled_out_rao"]:
        raise SystemExit(
            "not flat: wait for open bets to close and parked pool "
            "to settle, then evict")
    tx = client.evict(hotkey, keypair=keypair)
    return {"hotkey": hotkey, "status": "evicted", "tx": tx,
            "paid_to": info["refund_coldkey"],
            "alpha": info["balance_rao"] / RAO_PER_ALPHA}


def audit(client: CollateralClient, hotkey: str, from_block: int = 0) -> dict:
    """Prove from the event log that the owner never exceeded your
    bets.  Replays TradePosted / TradeClosed / TradeExpired for the
    hotkey and checks, per trade key: every debit stayed within the
    bet you posted, and nothing was ever debited that you did not
    post.  Violations should always be empty -- the contract enforces
    both; the audit exists so a miner can verify without trusting
    anything the owner runs."""
    hk32 = ss58_to_bytes32(hotkey)

    def logs(event):
        try:
            return event.get_logs(from_block=from_block,
                                  argument_filters={"hotkey": hk32})
        except TypeError:  # web3 < 7 kwarg spelling
            return event.get_logs(fromBlock=from_block,
                                  argument_filters={"hotkey": hk32})

    evs = []
    for name in ("TradePosted", "TradeClosed", "TradeExpired"):
        for ev in logs(getattr(client.contract.events, name)):
            evs.append((ev["blockNumber"], ev["logIndex"], name, ev["args"]))
    evs.sort(key=lambda e: (e[0], e[1]))

    bet: Dict[int, int] = {}
    violations: List[str] = []
    n_post = n_close = n_expire = 0
    posted_rao = collected_rao = 0
    for _b, _i, name, a in evs:
        key = int(a["tradeKey"])
        if name == "TradePosted":
            n_post += 1
            bet[key] = int(a["collateralRao"])
            posted_rao += int(a["collateralRao"])
        elif name == "TradeClosed":
            n_close += 1
            got = int(a["lossCollected"])
            collected_rao += got
            if key not in bet:
                violations.append(f"key {key}: debit with no posted bet")
            elif got > bet[key]:
                violations.append(
                    f"key {key}: collected {got} above the bet {bet[key]}")
        elif name == "TradeExpired":
            n_expire += 1
    return {
        "hotkey": hotkey, "ok": not violations, "violations": violations,
        "posted": n_post, "closed": n_close, "expired": n_expire,
        "posted_alpha": posted_rao / RAO_PER_ALPHA,
        "collected_alpha": collected_rao / RAO_PER_ALPHA,
    }


# ------------------------------------------------------------------- CLI

def _client(args, need_key: bool) -> CollateralClient:
    try:
        from config import FLOW_COLLATERAL_ADDRESS, FLOW_COLLATERAL_RPC
    except ImportError:
        FLOW_COLLATERAL_ADDRESS = ""
        FLOW_COLLATERAL_RPC = "https://lite.chain.opentensor.ai"
    rpc = (args.rpc or os.environ.get("FLOW_RPC")
           or os.environ.get("FLOW_COLLATERAL_RPC") or FLOW_COLLATERAL_RPC)
    contract = (args.contract or os.environ.get("FLOW_CONTRACT")
                or os.environ.get("FLOW_COLLATERAL_ADDRESS")
                or FLOW_COLLATERAL_ADDRESS)
    key = args.evm_key or os.environ.get("FLOW_EVM_KEY")
    if not rpc or not contract or contract.lower() in ("off", "none", "-"):
        raise SystemExit("need --rpc/--contract (or FLOW_RPC/FLOW_CONTRACT)")
    if need_key and not key:
        raise SystemExit("need --evm-key (or FLOW_EVM_KEY) to post")
    return CollateralClient(contract, rpc,
                       private_key=key if need_key else None)


def _keypair(args):
    if not args.wallet:
        return None
    return load_hotkey_keypair(args.wallet, args.wallet_hotkey,
                               args.wallet_path)


def _vector(path: str) -> List[float]:
    vec = json.load(open(path))
    if isinstance(vec, dict):               # payload {"FLOW": [...]}
        vec = next(iter(vec.values()))
    return [float(x) for x in vec]


def _parse_keys(items: List[str]) -> List[Tuple[int, int]]:
    out = []
    for it in items:
        reg, tid = it.split(":")
        out.append((REGIME_NAMES.index(reg.upper()), int(tid)))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rpc"), ap.add_argument("--contract")
    ap.add_argument("--evm-key",
                    help="EVM key; with --wallet it only pays gas")
    ap.add_argument("--wallet",
                    help="bittensor wallet name: sign each bet with "
                         "your hotkey (the native path)")
    ap.add_argument("--wallet-hotkey", default="default")
    ap.add_argument("--wallet-path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fnd = sub.add_parser("fund",
                         help="open the slot (hotkey-signed) or top it up")
    fnd.add_argument("--hotkey", required=True)
    fnd.add_argument("--alpha", type=float, required=True,
                     help="alpha to add (minimum 1 to open)")
    fnd.add_argument("--refund-coldkey",
                     help="ss58 that receives withdrawals; fixed at "
                          "open, defaults to the EVM key's mirror")

    ev = sub.add_parser("evict",
                        help="hotkey-signed force-exit of a flat book")
    ev.add_argument("--hotkey", required=True)

    bk = sub.add_parser("book", help="read the position from chain")
    bk.add_argument("--hotkey", required=True)

    p = sub.add_parser("post", help="post your bets on-chain")
    p.add_argument("--hotkey", required=True)
    p.add_argument("--vector-json", help="file with the dim-28 vector")
    p.add_argument("--regime", help="single bet: regime letter A-D")
    p.add_argument("--trade-id", type=int)
    p.add_argument("--f", type=float, help="kelly fraction for --regime")
    p.add_argument("--collateral-alpha", type=float,
                   help="override the on-chain collateral read")

    s = sub.add_parser("status", help="read your bets back from chain")
    s.add_argument("--hotkey", required=True)
    s.add_argument("--keys", nargs="+", required=True,
                   help="REGIME:TRADE_ID, e.g. C:7 D:2")

    a = sub.add_parser("audit", help="prove no debit exceeded your bets")
    a.add_argument("--hotkey", required=True)
    a.add_argument("--from-block", type=int, default=0)

    args = ap.parse_args(argv)
    if args.cmd == "fund":
        out = fund_position(_client(args, need_key=True), args.hotkey,
                            args.alpha, _keypair(args),
                            args.refund_coldkey)
    elif args.cmd == "evict":
        out = evict_position(_client(args, need_key=True), args.hotkey,
                             _keypair(args))
    elif args.cmd == "book":
        out = book(_client(args, need_key=False), args.hotkey)
    elif args.cmd == "post":
        client = _client(args, need_key=True)
        kp = _keypair(args)
        if args.vector_json:
            out = post_for_vector(client, args.hotkey,
                                  _vector(args.vector_json), kp,
                                  args.collateral_alpha)
        else:
            if not (args.regime and args.trade_id and args.f):
                raise SystemExit(
                    "need --vector-json or --regime/--trade-id/--f")
            collateral = (args.collateral_alpha if args.collateral_alpha is not None
                     else client.active_collateral_of(args.hotkey))
            out = post_bet(client, args.hotkey,
                           REGIME_NAMES.index(args.regime.upper()),
                           args.trade_id, args.f * collateral, kp)
    elif args.cmd == "status":
        client = _client(args, need_key=False)
        out = {"book": book(client, args.hotkey),
               "bets": status(client, args.hotkey, _parse_keys(args.keys))}
    else:
        out = audit(_client(args, False), args.hotkey, args.from_block)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
