"""Client for the FLOW collateral pool on the Bittensor EVM.

Real skin in the game: miners deposit subnet alpha behind their hotkey
(subnet 123 alpha, amounts in rao: 1 alpha = 1e9 rao, minimum
1 alpha, no maximum).  Live, the validator prices every trade by the
bet posted on chain for that trade (see flow.py) — idle balance is
not skin.  Once per settlement period the alpha lost on losing trades
is distributed pro rata to the winners; the contract enforces the
batch zero-sum, so settlement can only move alpha from losers to
winners.

Custody: the collateral is held on chain, owned by the contract's mirrored
ss58 coldkey via the staking-v2 precompile, delegated to the miner's own
hotkey.  Only a coldkey can add collateral by construction —
`transferStakeFrom` pulls from the caller's own mirror coldkey — and
only the hotkey can say who runs its slot (`addCollateralSigned`), so
one coldkey can run positions behind any number of hotkeys, each
independent, and nobody can squat a registered hotkey by funding it
first.

THE MODEL: miners post their own bets, by hand, on chain -- the owner
can only resolve them later.  The settlement daemon (team tooling,
kept and run outside this repo, next to the owner keys) decrypts
every payload at arrival through the envelope's owner leg, so
resolution is live; the validator in this repo stays public-only:

  post_trade / post_trade_signed
               MINER: reserve the bet's worst case (normally Kelly
               fraction x collateral) behind its trade key, before
               the payload that carries it uploads.  Immutable once
               posted; expiry derives on-chain from the regime in the
               key.  The signed path is authored by the HOTKEY itself
               (sr25519, verified by the chain at 0x...0403) and can
               be relayed by any gas payer without gaining authority.
  close_batch  OWNER: when trades resolve, post realized losses; each
               debits immediately into that settlement period's pool,
               bounded by the miner's own bet (wins and flats close
               with loss 0); no debit into a week that has not started
  settle       OWNER: once per period -- and only after the week ends
               on the chain clock -- pay the accumulated pool to the
               period's net winners pro rata after refunding
               over-debited books (zero-sum, period ids strictly
               ordered)

Withdrawals are single-step and instant, gated by margin, not time:
the miner's own posted bets reserve their summed worst case and
everything above the reserve leaves in one transaction.  If the owner
goes silent, reserves self-release at their on-chain expiry
(`sweepExpired` is permissionless), so no owner failure can lock a
position.

Custody flow (miner side):
  1. move alpha (delegated to your hotkey, on subnet 123) under
     your EVM key's mirror coldkey (substrate `transfer_stake` to the
     mirror ss58 of your H160, or `btcli evm stake`)
  2. `approve(pool_contract, netuid, amount)` on the staking-v2
     precompile (0x...0805) -- `add_collateral` below does this for you
  3. `addCollateralSigned(hotkey, amount, refundColdkey, nonce, r, s)`:
     YOUR HOTKEY signs the funder and refund coldkey in (nobody can
     squat your slot by funding it first); the contract pulls the
     alpha.  Top-ups after that are plain depositor-only
     `addCollateral`.  `add_collateral(keypair=...)` below does both.
  4. `post_trade` (or `post_trade_signed` with your hotkey) per bet,
     alongside each payload upload -- flow_post.py does this for you

Recovery: `evict(hotkey, nonce, r, s)` -- hotkey-signed, relayable --
force-exits a FLAT position to its RECORDED refund coldkey and frees
the slot (lost depositor key, or unwinding a position created against
the hotkey owner's wishes; it cannot redirect a rao, whoever signs).

The on-chain pool is FlowCollateralPool.  This file is the client;
the Solidity sources are not in the validator archive.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MAINNET_RPC = "https://archive.chain.opentensor.ai"
TESTNET_RPC = "https://test.chain.opentensor.ai"

# FlowCollateralPool deploy block; log scans never need to look earlier.
DEPLOY_BLOCK = 8859422
# archive rejects wide eth_getLogs ranges (prohibited_shape); chunk to this.
GETLOGS_CHUNK = 500

# lite.chain.opentensor.ai 429s under burst reads (deploy receipt
# polls, getLogs, eth_call).  Retry only transient transport; a
# revert / bad nonce / insufficient gas is not a fetch failure.
RPC_TIMEOUT_S = float(os.environ.get("FLOW_RPC_TIMEOUT", "45"))
RPC_RETRY_TRIES = int(os.environ.get("FLOW_RPC_RETRIES", "8"))
RPC_RETRY_BASE_S = float(os.environ.get("FLOW_RPC_RETRY_BASE", "1"))
RPC_RETRY_MAX_S = float(os.environ.get("FLOW_RPC_RETRY_MAX", "20"))


def is_transient_rpc(exc: BaseException) -> bool:
    """True for rate-limits, timeouts, and dropped connections."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in ("Timeout", "ReadTimeout", "ConnectTimeout", "TimeExhausted",
                "ConnectionError", "ConnectTimeoutError", "ProtocolError"):
        return True
    if "429" in msg or "too many requests" in msg:
        return True
    if any(s in msg for s in (
            "timeout", "timed out", "connection reset", "connection aborted",
            "connection refused", "temporarily unavailable", "bad gateway",
            "service unavailable", "gateway timeout", "502", "503", "504")):
        return True
    try:
        import requests
        if isinstance(exc, requests.exceptions.Timeout):
            return True
        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code in (429, 500, 502, 503, 504)
    except Exception:  # noqa: BLE001 - requests may be absent
        pass
    return False


def rpc_retry(fn, *, tries: int = RPC_RETRY_TRIES, what: str = "rpc"):
    """Call `fn` until it succeeds or the error is not transient."""
    delay = RPC_RETRY_BASE_S
    last: Optional[BaseException] = None
    for i in range(max(tries, 1)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - RPC layer raises many types
            last = e
            if not is_transient_rpc(e) or i == tries - 1:
                raise
            logger.warning("%s transient (%s: %s); retry %d/%d in %.0fs",
                           what, type(e).__name__, e, i + 1, tries, delay)
            time.sleep(delay)
            delay = min(delay * 2.0, RPC_RETRY_MAX_S)
    raise last  # pragma: no cover


def make_web3(rpc_url: str, timeout: float = RPC_TIMEOUT_S):
    """HTTP web3 whose every JSON-RPC call retries 429/timeout/5xx."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
    inner = w3.provider.make_request

    def _wrapped(method, params):
        return rpc_retry(lambda: inner(method, params), what=method)

    w3.provider.make_request = _wrapped  # type: ignore[method-assign]
    return w3


def wait_receipt(w3, txh, timeout: float = 180.0):
    """Poll for a receipt.  A 429 mid-wait is not a failure — keep polling."""
    deadline = time.time() + timeout
    delay = 1.0
    hx = txh.hex() if hasattr(txh, "hex") else str(txh)
    while time.time() < deadline:
        try:
            rcpt = w3.eth.get_transaction_receipt(txh)
            if rcpt is not None and rcpt.get("blockNumber"):
                return rcpt
        except Exception as e:  # noqa: BLE001
            not_found = (type(e).__name__ == "TransactionNotFound"
                         or "not found" in str(e).lower())
            if not (not_found or is_transient_rpc(e)):
                raise
            if is_transient_rpc(e):
                logger.warning("receipt poll %s: %s", hx, e)
        time.sleep(delay)
        delay = min(delay * 1.5, 8.0)
    raise TimeoutError(f"timed out waiting for {hx} — the tx may still "
                       "be in the mempool; do not re-sign a new nonce "
                       "until you have checked the explorer")

RAO_PER_ALPHA = 10**9

# staking-v2 precompile: alpha custody + coldkey-gated transfers
ALPHA_PRECOMPILE_ADDRESS = "0x0000000000000000000000000000000000000805"
SR25519_VERIFY_ADDRESS = "0x0000000000000000000000000000000000000403"
ALPHA_PRECOMPILE_ABI = [
    {"type": "function", "name": "approve", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"},
                {"name": "netuid", "type": "uint256"},
                {"name": "amount", "type": "uint256"}], "outputs": []},
    {"type": "function", "name": "allowance", "stateMutability": "view",
     "inputs": [{"name": "approver", "type": "address"},
                {"name": "spender", "type": "address"},
                {"name": "netuid", "type": "uint256"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "getStake", "stateMutability": "view",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "coldkey", "type": "bytes32"},
                {"name": "netuid", "type": "uint256"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]


def ss58_to_bytes32(key: str) -> bytes:
    """ss58 address (or 0x-hex pubkey) -> 32-byte public key."""
    if key.startswith("0x"):
        raw = bytes.fromhex(key[2:])
    else:
        try:                       # bittensor <= 10 ships scalecodec
            from scalecodec.utils.ss58 import ss58_decode
            raw = bytes.fromhex(ss58_decode(key))
        except ModuleNotFoundError:  # bittensor 11: the Rust core
            from bittensor_core import ss58_decode
            raw = bytes(ss58_decode(key))
    if len(raw) != 32:
        raise ValueError(f"key does not decode to 32 bytes: {key}")
    return raw


def hotkey_id(hotkey: str) -> str:
    """Canonical 0x-hex pubkey for daemon refs and event joins.

    The datalog and miner tooling speak ss58; contract logs emit the
    raw 32-byte pubkey.  One form here so a posted bet and its panel
    trade always hash to the same key.
    """
    return "0x" + ss58_to_bytes32(hotkey).hex()


def evm_mirror_coldkey(address: str) -> bytes:
    """H160 -> mirrored ss58 pubkey: blake2_256(b"evm:" ++ address_bytes)."""
    raw = bytes.fromhex(address[2:] if address.startswith("0x") else address)
    if len(raw) != 20:
        raise ValueError(f"not an H160 address: {address}")
    return hashlib.blake2b(b"evm:" + raw, digest_size=32).digest()


def trade_key(regime: int, trade_id: int) -> int:
    """Contract trade key: (regime << 32) | trade_id.

    The wire trade id is unique per regime; packing the regime index in
    the high bits makes the key unique per hotkey, which the contract
    requires (one open per key)."""
    return (int(regime) << 32) | (int(trade_id) & 0xFFFFFFFF)


# Snapshot pinning: collateral reads are made at the newest block height that
# is a multiple of this, so every validator sampling within the same
# window reads the identical chain state.  300 blocks = one hour at 12s.
PIN_BLOCKS = 300

# Free-look guard: a bet must be on chain within this of its trade's
# open or it carries no money in either direction (forgives pipeline
# lag, not tape-watching).  One constant shared by the settlement
# daemon (losses) and the validator's emission weighting (pay), so a
# late bet is void everywhere at once.
LATE_POST_GRACE_S = 900.0


def make_bet_collateral_fn(bets: Dict[str, Tuple[float, Optional[float]]],
                      *, chain_ts: float, now_hour: float,
                      grace_s: float = LATE_POST_GRACE_S):
    """Per-trade collateral pricer: THE POSTED BET is the only money.

    `bets` maps DaemonState-style refs (`hotkey_id(hk)|trade_key`) to
    (bet_alpha, posted_ts).  The returned callable matches the
    `collateral_fn(hk, open_hour, regime, tid, f)` protocol of
    flow.compute_collateral_settlement / compute_flow_salience and returns
    the implied collateral `bet / f`, so every downstream `x f`
    multiplication lands back on the bet itself:

        loss  = max(-R, 0) x f x (bet/f) = max(-R, 0) x bet
        claim = max(+R, 0) x f x (bet/f) = max(+R, 0) x bet
        pay  ~=            score x (bet-sized skin)

    Symmetric by construction: a bet the miner never posted prices
    every leg at zero, and no deposit or withdrawal after the open can
    move either side (the bet is immutable chain state posted before
    the open).  A bet posted later than `grace_s` after its trade's
    open is void both ways (free-look).  Open times are panel hours;
    they anchor to chain time through (`chain_ts`, `now_hour`) exactly
    like the daemon's close leg, so both legs void the same bets.
    """
    def _fn(hk: str, open_hour: float, regime: int = None,
            tid: int = None, f: float = None) -> float:
        if regime is None or tid is None or not f or f <= 0.0:
            return 0.0
        bet, posted_ts = bets.get(
            f"{hotkey_id(hk)}|{trade_key(regime, tid)}", (0.0, None))
        if bet <= 0.0:
            return 0.0
        open_ts = chain_ts + (open_hour - now_hour) * 3600.0
        if posted_ts is not None and posted_ts > open_ts + grace_s:
            return 0.0
        return bet / f
    return _fn


def pro_rata_payouts(pool_rao: int,
                     claims_alpha: Dict[str, float]) -> Dict[str, int]:
    """Split an integer rao pool pro rata by claim, exactly zero-sum.

    Floor division per winner; the rounding remainder goes to the
    largest claim.  Pure function so the posted-batch arithmetic is
    testable and recomputable without a chain connection.
    """
    total = sum(c for c in claims_alpha.values() if c > 0)
    if pool_rao <= 0 or total <= 0:
        return {}
    winners = sorted((hk for hk, c in claims_alpha.items() if c > 0),
                     key=lambda hk: (-claims_alpha[hk], hk))
    pays = {hk: int(pool_rao * claims_alpha[hk] // total) for hk in winners}
    pays[winners[0]] += pool_rao - sum(pays.values())
    return {hk: p for hk, p in pays.items() if p > 0}


def compose_settle_payouts(pool_rao: int,
                           claims_alpha: Dict[str, float],
                           refunds_alpha: Optional[Dict[str, float]] = None
                           ) -> Dict[str, int]:
    """One settle batch: over-collection refunds plus net claims pro rata.

    Attribution follows flow.compute_collateral_settlement, which settles NET
    per hotkey per period: a book's wins offset its losses.  The live
    feed necessarily debits gross per losing trade (a close cannot know
    about later wins in the period), so at settle each book is first
    paid back what it was over-debited (gross collected minus its net
    loss), and only the remaining net pool splits pro rata by net claim.
    A hotkey can hold both a refund and a claim; the amounts merge into
    one winner entry.  Exactly zero-sum in rao: refunds are clamped to
    the pool and the pro-rata split of the remainder is exact, so the
    contract's NotZeroSum check holds.
    """
    refunds = {hk: int(round(a * RAO_PER_ALPHA))
               for hk, a in (refunds_alpha or {}).items() if a > 0}
    refunds = {hk: r for hk, r in refunds.items() if r > 0}
    total_r = sum(refunds.values())
    if pool_rao <= 0:
        return {}
    if total_r > pool_rao:
        # rounding or on-chain clamps say the pool is smaller than the
        # over-collection; scale down, largest refund takes the remainder
        scaled = {hk: int(pool_rao * r // total_r) for hk, r in refunds.items()}
        order = sorted(refunds, key=lambda hk: (-refunds[hk], hk))
        scaled[order[0]] += pool_rao - sum(scaled.values())
        refunds = {hk: r for hk, r in scaled.items() if r > 0}
        total_r = sum(refunds.values())
    pays = pro_rata_payouts(pool_rao - total_r, claims_alpha)
    for hk, r in refunds.items():
        pays[hk] = pays.get(hk, 0) + r
    return {hk: p for hk, p in pays.items() if p > 0}


FLOW_COLLATERAL_ABI = [
    {"type": "constructor", "inputs": [
        {"name": "_netuid", "type": "uint256"},
        {"name": "_periodZero", "type": "uint64"}]},
    {"type": "event", "name": "CollateralAdded", "anonymous": False,
     "inputs": [{"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "depositor", "type": "address", "indexed": True},
                {"name": "amount", "type": "uint256", "indexed": False},
                {"name": "balance", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "TradePosted", "anonymous": False,
     "inputs": [{"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "tradeKey", "type": "uint64", "indexed": True},
                {"name": "collateralRao", "type": "uint256", "indexed": False},
                {"name": "expiry", "type": "uint64", "indexed": False}]},
    {"type": "function", "name": "postTrade",
     "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "tradeKey", "type": "uint64"},
                {"name": "collateralRao", "type": "uint256"}], "outputs": []},
    {"type": "function", "name": "postTradeSigned",
     "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "tradeKey", "type": "uint64"},
                {"name": "collateralRao", "type": "uint256"},
                {"name": "nonce", "type": "uint64"},
                {"name": "r", "type": "bytes32"},
                {"name": "s", "type": "bytes32"}], "outputs": []},
    {"type": "function", "name": "postNonce", "stateMutability": "view",
     "inputs": [{"name": "", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "uint64"}]},
    {"type": "function", "name": "periodZero", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint64"}]},
    {"type": "function", "name": "currentPeriod", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint64"}]},
    {"type": "function", "name": "PERIOD_SECONDS", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "event", "name": "TradeClosed", "anonymous": False,
     "inputs": [{"name": "periodId", "type": "uint64", "indexed": True},
                {"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "tradeKey", "type": "uint64", "indexed": True},
                {"name": "lossPosted", "type": "uint256", "indexed": False},
                {"name": "lossCollected", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "TradeExpired", "anonymous": False,
     "inputs": [{"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "tradeKey", "type": "uint64", "indexed": True},
                {"name": "exposure", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "Settled", "anonymous": False,
     "inputs": [{"name": "periodId", "type": "uint64", "indexed": True},
                {"name": "pool", "type": "uint256", "indexed": False},
                {"name": "winners", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "PoolRolled", "anonymous": False,
     "inputs": [{"name": "fromPeriod", "type": "uint64", "indexed": True},
                {"name": "toPeriod", "type": "uint64", "indexed": True},
                {"name": "amount", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "SettleWin", "anonymous": False,
     "inputs": [{"name": "periodId", "type": "uint64", "indexed": True},
                {"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "amount", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "Withdrawn", "anonymous": False,
     "inputs": [{"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "refundColdkey", "type": "bytes32", "indexed": False},
                {"name": "paid", "type": "uint256", "indexed": False},
                {"name": "sent", "type": "uint256", "indexed": False}]},
    {"type": "event", "name": "Reconciled", "anonymous": False,
     "inputs": [{"name": "fromHotkey", "type": "bytes32", "indexed": True},
                {"name": "toHotkey", "type": "bytes32", "indexed": True},
                {"name": "amount", "type": "uint256", "indexed": False}]},
    {"type": "function", "name": "addCollateral", "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "amount", "type": "uint256"},
                {"name": "refundColdkey", "type": "bytes32"}], "outputs": []},
    {"type": "function", "name": "addCollateralSigned",
     "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "amount", "type": "uint256"},
                {"name": "refundColdkey", "type": "bytes32"},
                {"name": "nonce", "type": "uint64"},
                {"name": "r", "type": "bytes32"},
                {"name": "s", "type": "bytes32"}], "outputs": []},
    {"type": "function", "name": "evict", "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "nonce", "type": "uint64"},
                {"name": "r", "type": "bytes32"},
                {"name": "s", "type": "bytes32"}], "outputs": []},
    {"type": "event", "name": "Evicted", "anonymous": False,
     "inputs": [{"name": "hotkey", "type": "bytes32", "indexed": True},
                {"name": "refundColdkey", "type": "bytes32", "indexed": False},
                {"name": "paid", "type": "uint256", "indexed": False},
                {"name": "sent", "type": "uint256", "indexed": False}]},
    {"type": "function", "name": "withdraw", "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "amount", "type": "uint256"}], "outputs": []},
    {"type": "function", "name": "sweepExpired", "stateMutability": "nonpayable",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "tradeKeys", "type": "uint64[]"}], "outputs": []},
    {"type": "function", "name": "closeBatch", "stateMutability": "nonpayable",
     "inputs": [{"name": "periodId", "type": "uint64"},
                {"name": "hotkeys", "type": "bytes32[]"},
                {"name": "tradeKeys", "type": "uint64[]"},
                {"name": "lossRao", "type": "uint256[]"}], "outputs": []},
    {"type": "function", "name": "settle", "stateMutability": "nonpayable",
     "inputs": [{"name": "periodId", "type": "uint64"},
                {"name": "winners", "type": "bytes32[]"},
                {"name": "winRao", "type": "uint256[]"}], "outputs": []},
    {"type": "function", "name": "reconcile", "stateMutability": "nonpayable",
     "inputs": [{"name": "fromHotkey", "type": "bytes32"},
                {"name": "toHotkey", "type": "bytes32"},
                {"name": "amount", "type": "uint256"}], "outputs": []},
    {"type": "function", "name": "setContractColdkey", "stateMutability": "nonpayable",
     "inputs": [{"name": "coldkey", "type": "bytes32"}], "outputs": []},
    {"type": "function", "name": "transferOwnership", "stateMutability": "nonpayable",
     "inputs": [{"name": "to", "type": "address"}], "outputs": []},
    {"type": "function", "name": "collateralOf", "stateMutability": "view",
     "inputs": [{"name": "hotkey", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "activeCollateralOf", "stateMutability": "view",
     "inputs": [{"name": "hotkey", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "freeCollateralOf", "stateMutability": "view",
     "inputs": [{"name": "hotkey", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "tradeInfo", "stateMutability": "view",
     "inputs": [{"name": "hotkey", "type": "bytes32"},
                {"name": "tradeKey", "type": "uint64"}],
     "outputs": [{"name": "exposure", "type": "uint256"},
                 {"name": "expiry", "type": "uint256"}]},
    {"type": "function", "name": "positionInfo", "stateMutability": "view",
     "inputs": [{"name": "hotkey", "type": "bytes32"}],
     "outputs": [{"name": "depositor", "type": "address"},
                 {"name": "balance", "type": "uint256"},
                 {"name": "firstFundedAt", "type": "uint256"},
                 {"name": "refundColdkey", "type": "bytes32"},
                 {"name": "openExposure", "type": "uint256"},
                 {"name": "openCount", "type": "uint256"},
                 {"name": "pooledOut", "type": "uint256"}]},
    {"type": "function", "name": "poolAccum", "stateMutability": "view",
     "inputs": [{"name": "", "type": "uint64"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "totalCollateral", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "lastSettledPeriod", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint64"}]},
    {"type": "function", "name": "MAX_OPEN_SECONDS", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "netuid", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "owner", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "address"}]},
]


class CollateralClient:
    """Thin web3 wrapper around the FlowCollateralPool contract.

    `collateral_map` is fail-static: on RPC failure the validator keeps
    weighting with the previous snapshot instead of zeroing every
    miner's skin.
    """

    def __init__(self, contract_address: str,
                 rpc_url: str = MAINNET_RPC,
                 private_key: Optional[str] = None):
        from web3 import Web3
        self.w3 = make_web3(rpc_url)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=FLOW_COLLATERAL_ABI)
        self.custody = self.w3.eth.contract(
            address=Web3.to_checksum_address(ALPHA_PRECOMPILE_ADDRESS),
            abi=ALPHA_PRECOMPILE_ABI)
        self._acct = (self.w3.eth.account.from_key(private_key)
                      if private_key else None)
        self._last_collateral: Dict[str, float] = {}
        self._last_ok_ts: float = 0.0
        # incremental TradePosted cache for bets_map (append-only)
        self._bets: Dict[str, Tuple[float, Optional[float]]] = {}
        self._bets_synced: int = DEPLOY_BLOCK
        self._block_ts: Dict[int, float] = {}

    # ------------------------------------------------------------- read API

    def netuid(self) -> int:
        return int(self.contract.functions.netuid().call())

    def collateral_of(self, hotkey: str) -> float:
        """Held alpha booked to `hotkey` (alpha units)."""
        rao = int(self.contract.functions.collateralOf(
            ss58_to_bytes32(hotkey)).call())
        return rao / RAO_PER_ALPHA

    def active_collateral_of(self, hotkey: str, block=None) -> float:
        """The skin the validator weights scores with (alpha units).
        Same as the booked balance: exposure is priced per trade at its
        open, so there is no pending tier."""
        fn = self.contract.functions.activeCollateralOf(ss58_to_bytes32(hotkey))
        rao = int(fn.call(block_identifier=block) if block is not None
                  else fn.call())
        return rao / RAO_PER_ALPHA

    def free_collateral_of(self, hotkey: str) -> float:
        """Withdrawable right now (alpha): balance minus the reserve
        held by open trades."""
        rao = int(self.contract.functions.freeCollateralOf(
            ss58_to_bytes32(hotkey)).call())
        return rao / RAO_PER_ALPHA

    def trade_info(self, hotkey: str, key: int) -> dict:
        exposure, expiry = self.contract.functions.tradeInfo(
            ss58_to_bytes32(hotkey), int(key)).call()
        return dict(exposure_rao=int(exposure), expiry=int(expiry))

    def post_nonce(self, hotkey: str) -> int:
        """Last accepted nonce of a hotkey-signed bet post."""
        return int(self.contract.functions.postNonce(
            ss58_to_bytes32(hotkey)).call())

    def position_info(self, hotkey: str) -> dict:
        (dep, bal, first, refund_ck, open_exp, open_cnt,
         pooled_out) = self.contract.functions.positionInfo(
            ss58_to_bytes32(hotkey)).call()
        return dict(depositor=dep, balance_rao=int(bal),
                    first_funded_at=int(first),
                    refund_coldkey="0x" + bytes(refund_ck).hex(),
                    open_exposure_rao=int(open_exp),
                    open_count=int(open_cnt),
                    pooled_out_rao=int(pooled_out))

    def pool_of(self, period_id: int) -> int:
        """Accumulated pool for a period (rao)."""
        return int(self.contract.functions.poolAccum(int(period_id)).call())

    def last_settled_period(self) -> int:
        return int(self.contract.functions.lastSettledPeriod().call())

    def period_zero(self) -> int:
        """Challenge genesis timestamp: period 1 opens here (chain state)."""
        return int(self.contract.functions.periodZero().call())

    def current_period(self) -> int:
        """The settlement week the chain clock is in right now (0 =
        before genesis).  Period k settles only once currentPeriod > k."""
        return int(self.contract.functions.currentPeriod().call())

    def pinned_block(self) -> int:
        """Newest block height that is a multiple of PIN_BLOCKS."""
        n = int(self.w3.eth.block_number)
        return max(n - (n % PIN_BLOCKS), 1)

    def collateral_map(self, hotkeys: Iterable[str],
                   block: Optional[int] = None) -> Dict[str, float]:
        """Held alpha for a batch of hotkeys (alpha units).

        Block-pinned: all reads target the same rounded block height
        (`pinned_block`, or an explicit `block` such as a trade's open
        block), so every validator sampling within the same window
        computes weights from the identical snapshot.  If the RPC does
        not serve historical state, falls back to latest.  Fail-static:
        on RPC failure, serve the last successful snapshot for known
        keys (0.0 for unknown ones).
        """
        hks = list(hotkeys)
        try:
            try:
                blk = self.pinned_block() if block is None else int(block)
                out = {hk: self.active_collateral_of(hk, block=blk)
                       for hk in hks}
            except Exception:  # noqa: BLE001 - pruned node, no history
                out = {hk: self.active_collateral_of(hk) for hk in hks}
            self._last_collateral = dict(out)
            self._last_ok_ts = time.time()
            return out
        except Exception as e:  # noqa: BLE001 - RPC layer raises many types
            logger.warning("collateral RPC failed (%s); serving last snapshot "
                           "(%.0fs old)", e, time.time() - self._last_ok_ts)
            return {hk: self._last_collateral.get(hk, 0.0) for hk in hks}

    def bets_map(self) -> Dict[str, Tuple[float, Optional[float]]]:
        """Every bet ever posted, from TradePosted event logs.

        Returns {`hotkey_id|trade_key`: (bet_alpha, posted_ts)} —
        the immutable pre-open commitments that price all money and
        all emission skin (see make_bet_collateral_fn).  Incremental and
        append-only: each call scans from the last synced block and
        merges.  Fail-static like collateral_map: on RPC failure the last
        snapshot is served rather than zeroing every miner's skin.
        """
        try:
            tip = int(self.w3.eth.block_number)
            start = min(self._bets_synced, tip)
            ev = self.contract.events.TradePosted
            logs = []
            for lo in range(start, tip + 1, GETLOGS_CHUNK):
                hi = min(lo + GETLOGS_CHUNK - 1, tip)
                try:
                    logs.extend(ev.get_logs(from_block=lo, to_block=hi))
                except TypeError:   # web3 < 7 spells the kwargs fromBlock
                    logs.extend(ev.get_logs(fromBlock=lo, toBlock=hi))
                if hi < tip:
                    time.sleep(0.25)    # pace the backfill; archive 429s bursts
            for lg in logs:
                ref = ("0x" + bytes(lg["args"]["hotkey"]).hex()
                       + f"|{int(lg['args']['tradeKey'])}")
                if ref in self._bets:
                    continue    # immutable: first sighting wins
                bn = int(lg["blockNumber"])
                if bn not in self._block_ts:
                    self._block_ts[bn] = float(
                        self.w3.eth.get_block(bn)["timestamp"])
                self._bets[ref] = (
                    int(lg["args"]["collateralRao"]) / RAO_PER_ALPHA,
                    self._block_ts[bn])
            self._bets_synced = tip
        except Exception as e:  # noqa: BLE001 - RPC layer raises many types
            logger.warning("bet log scan failed (%s); serving %d cached "
                           "bets", e, len(self._bets))
        if not self._bets:
            logger.warning(
                "bets_map is EMPTY after a full scan: every FLOW trade will "
                "be priced at zero skin. If miners have posted bets, this "
                "validator's RPC is likely pruning logs — use an archive "
                "endpoint (FLOW_COLLATERAL_RPC).")
        return dict(self._bets)

    # ------------------------------------------------------------ write API

    def _send(self, fn, gas: int = 3_000_000) -> str:
        if self._acct is None:
            raise RuntimeError("CollateralClient constructed without a private key")
        # nonce/gas/chainId go through the retrying provider.  The
        # signed payload is broadcast once; a 429 after the node
        # accepted it is recovered by polling the same hash, never by
        # bumping the nonce (that would be a second tx).
        tx = fn.build_transaction({
            "from": self._acct.address,
            "nonce": self.w3.eth.get_transaction_count(self._acct.address),
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
            "chainId": self.w3.eth.chain_id,
        })
        signed = self._acct.sign_transaction(tx)
        raw = signed.raw_transaction
        known = getattr(signed, "hash", None)
        try:
            txh = self.w3.eth.send_raw_transaction(raw)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if known is not None and (
                    "already known" in msg or "known transaction" in msg
                    or "nonce too low" in msg):
                logger.warning("broadcast raced (%s); waiting on %s",
                               e, known.hex())
                txh = known
            else:
                raise
        rcpt = wait_receipt(self.w3, txh, timeout=180)
        if rcpt["status"] != 1:
            hx = txh.hex() if hasattr(txh, "hex") else str(txh)
            raise RuntimeError(f"tx reverted: {hx}")
        return txh.hex() if hasattr(txh, "hex") else str(txh)

    def fund_digest(self, hotkey: str, funder_address: str,
                    refund_coldkey_b32: bytes, amount_rao: int,
                    nonce: int) -> bytes:
        """The 32-byte message the HOTKEY signs to open its position:
        keccak256("FLOWFUND" ++ chainid ++ contract ++ hotkey ++
        funder ++ refundColdkey ++ amount ++ nonce).  Only the hotkey
        can say who its depositor and refund coldkey are, so nobody
        can squat a registered hotkey's slot by funding it first."""
        from web3 import Web3
        return bytes(Web3.solidity_keccak(
            ["string", "uint256", "address", "bytes32", "address",
             "bytes32", "uint256", "uint64"],
            ["FLOWFUND", int(self.w3.eth.chain_id), self.contract.address,
             ss58_to_bytes32(hotkey), funder_address,
             bytes(refund_coldkey_b32), int(amount_rao), int(nonce)]))

    def evict_digest(self, hotkey: str, nonce: int) -> bytes:
        """The 32-byte message the HOTKEY signs to force-exit its flat
        position back to its own recorded refund coldkey (recovery /
        unsquat; cannot redirect funds, whoever signs or relays)."""
        from web3 import Web3
        return bytes(Web3.solidity_keccak(
            ["string", "uint256", "address", "bytes32", "uint64"],
            ["FLOWEVICT", int(self.w3.eth.chain_id), self.contract.address,
             ss58_to_bytes32(hotkey), int(nonce)]))

    def add_collateral(self, hotkey: str, amount_alpha: float,
                  refund_coldkey: Optional[str] = None,
                  keypair=None, signature: Optional[bytes] = None,
                  nonce: Optional[int] = None) -> str:
        """Approve + add in sequence.

        Top-ups (position already open) need nothing extra.  OPENING a
        position requires the hotkey's own sr25519 authorization over
        (this funder, the refund coldkey, the amount): pass `keypair`
        (the bittensor hotkey keypair; signs locally) or a
        pre-computed 64-byte `signature` over `fund_digest(...)` with
        its `nonce`.  `refund_coldkey` (ss58 or 0x-hex) receives
        withdrawals; fixed at open, defaults to this account's own
        EVM-mirror coldkey."""
        rao = int(round(amount_alpha * RAO_PER_ALPHA))
        refund = (ss58_to_bytes32(refund_coldkey) if refund_coldkey
                  else evm_mirror_coldkey(self._acct.address))
        self._send(self.custody.functions.approve(
            self.contract.address, self.netuid(), rao))
        if int(self.position_info(hotkey)["depositor"], 16) != 0:
            return self._send(self.contract.functions.addCollateral(
                ss58_to_bytes32(hotkey), rao, refund))
        if signature is None:
            if keypair is None:
                raise ValueError(
                    "opening a position needs the hotkey's signature: "
                    "pass keypair= (or signature= + nonce=)")
            if bytes(keypair.public_key) != ss58_to_bytes32(hotkey):
                raise ValueError("keypair does not own this hotkey")
            nonce = self.post_nonce(hotkey) + 1
            signature = bytes(keypair.sign(self.fund_digest(
                hotkey, self._acct.address, refund, rao, nonce)))
        if nonce is None:
            raise ValueError("signature= requires nonce=")
        sig = bytes(signature)
        if len(sig) != 64:
            raise ValueError("sr25519 signature must be 64 bytes")
        return self._send(self.contract.functions.addCollateralSigned(
            ss58_to_bytes32(hotkey), rao, refund, int(nonce),
            sig[:32], sig[32:]))

    def evict(self, hotkey: str, keypair=None,
              signature: Optional[bytes] = None,
              nonce: Optional[int] = None) -> str:
        """Force-exit a flat position to its recorded refund coldkey
        with the hotkey's signature (recovery / unsquat).  Anyone may
        relay; the payout destination is fixed on-chain."""
        if signature is None:
            if keypair is None:
                raise ValueError("evict needs keypair= or signature=")
            if bytes(keypair.public_key) != ss58_to_bytes32(hotkey):
                raise ValueError("keypair does not own this hotkey")
            nonce = self.post_nonce(hotkey) + 1
            signature = bytes(keypair.sign(self.evict_digest(hotkey, nonce)))
        if nonce is None:
            raise ValueError("signature= requires nonce=")
        sig = bytes(signature)
        if len(sig) != 64:
            raise ValueError("sr25519 signature must be 64 bytes")
        return self._send(self.contract.functions.evict(
            ss58_to_bytes32(hotkey), int(nonce), sig[:32], sig[32:]))

    def withdraw(self, hotkey: str, amount_alpha: float) -> str:
        """Withdraw immediately, up to the free collateral (balance minus the
        reserve held by open trades).  Single-step; no delay."""
        return self._send(self.contract.functions.withdraw(
            ss58_to_bytes32(hotkey),
            int(round(amount_alpha * RAO_PER_ALPHA))))

    def sweep_expired(self, hotkey: str, keys: Sequence[int]) -> str:
        """Release the reserve of trades whose expiry passed without a
        close (owner outage path).  Permissionless."""
        return self._send(self.contract.functions.sweepExpired(
            ss58_to_bytes32(hotkey), [int(k) for k in keys]))

    def post_trade(self, hotkey: str, key: int, collateral_alpha: float) -> str:
        """THE MODEL: the miner posts the bet, by hand, on chain
        (depositor path).  Reserves the bet's worst case -- normally
        kelly f x collateral -- behind its trade key; the owner can
        only resolve it later, never create, resize, or extend one.
        Immutable once posted; expiry derives on-chain from the regime
        in the key."""
        return self._send(self.contract.functions.postTrade(
            ss58_to_bytes32(hotkey), int(key),
            int(round(collateral_alpha * RAO_PER_ALPHA))))

    def post_digest(self, hotkey: str, key: int, collateral_rao: int,
                    nonce: int) -> bytes:
        """The 32-byte message the HOTKEY signs to post a bet:
        keccak256("FLOWPOST" ++ chainid ++ contract ++ hotkey ++
        tradeKey ++ collateralRao ++ nonce), byte-identical to what
        postTradeSigned computes -- binding the bet to this chain,
        this contract, this key, this exact size."""
        from web3 import Web3
        return bytes(Web3.solidity_keccak(
            ["string", "uint256", "address", "bytes32", "uint64",
             "uint256", "uint64"],
            ["FLOWPOST", int(self.w3.eth.chain_id), self.contract.address,
             ss58_to_bytes32(hotkey), int(key), int(collateral_rao),
             int(nonce)]))

    def post_trade_signed(self, hotkey: str, key: int, collateral_alpha: float,
                          nonce: int, signature: bytes) -> str:
        """Relay a hotkey-signed bet (anyone may call; this account
        only pays the gas).  `signature` is the raw 64-byte sr25519
        signature over `post_digest(...)`."""
        sig = bytes(signature)
        if len(sig) != 64:
            raise ValueError("sr25519 signature must be 64 bytes")
        return self._send(self.contract.functions.postTradeSigned(
            ss58_to_bytes32(hotkey), int(key),
            int(round(collateral_alpha * RAO_PER_ALPHA)), int(nonce),
            sig[:32], sig[32:]))

    def close_batch(self, period_id: int,
                    closes: Sequence[Tuple[str, int, float]]) -> Optional[str]:
        """Post resolved trades (owner only).

        `closes` items: (hotkey, trade_key, loss_alpha); wins and flats
        close with loss 0.  Losses debit immediately into `period_id`'s
        pool; the contract bounds each by the trade's posted exposure
        and clamps to the position.  Returns None for an empty batch.
        """
        if not closes:
            return None
        return self._send(self.contract.functions.closeBatch(
            int(period_id),
            [ss58_to_bytes32(hk) for hk, _, _ in closes],
            [int(k) for _, k, _ in closes],
            [int(round(l * RAO_PER_ALPHA)) for _, _, l in closes]))

    def settle_period(self, period_id: int,
                      claims_alpha: Dict[str, float],
                      refunds_alpha: Optional[Dict[str, float]] = None) -> str:
        """Pay one period's accumulated pool (owner only).

        The pool is read from the chain (`poolAccum`); the batch is
        composed by `compose_settle_payouts`: gross-over-net refunds
        first (the live feed debits per losing trade; the settlement
        attribution is net per hotkey, flow.compute_collateral_settlement),
        then the remaining net pool pro rata by net claim.  Exactly
        zero-sum, which the contract enforces.  With no positive
        payouts the settle posts empty and the contract rolls the pool
        into the next period.

        A payout whose position no longer exists is dropped before the
        split: a winner who fully exited between resolution and settle
        (a flat full-exit deletes the position) would make the contract
        revert NoPosition, and with period ids strictly ordered that
        one exit would block this settle and every later period.  A
        dropped share redistributes pro rata over the remaining
        winners; if no one is left standing the settle posts empty and
        the pool rolls.  Deterministic from chain state at the settle
        block, so the batch stays recomputable.
        """
        pool = self.pool_of(period_id)

        def _live(m: Optional[Dict[str, float]], what: str) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for hk, v in (m or {}).items():
                if v <= 0:
                    continue
                if int(self.position_info(hk)["depositor"], 16) == 0:
                    logger.warning(
                        "settle %d: dropping %s of %s (position fully "
                        "exited); its share redistributes pro rata",
                        period_id, what, hk)
                    continue
                out[hk] = v
            return out

        pays = compose_settle_payouts(pool,
                                      _live(claims_alpha, "claim"),
                                      _live(refunds_alpha, "refund"))
        if pays and sum(pays.values()) != pool:
            # refunds with no claimant left cannot absorb the whole
            # pool; roll rather than revert NotZeroSum on-chain
            logger.warning(
                "settle %d: composed payouts %d != pool %d rao; posting "
                "an empty settle (roll) instead",
                period_id, sum(pays.values()), pool)
            pays = {}
        w_keys = sorted(pays)
        return self._send(self.contract.functions.settle(
            int(period_id),
            [ss58_to_bytes32(hk) for hk in w_keys],
            [pays[hk] for hk in w_keys]))

    def reconcile(self, from_hotkey: str, to_hotkey: str,
                  amount_rao: int) -> str:
        """Repair physical drift (owner only): move contract-owned collateral
        from a hotkey holding more than its booked total to one holding
        less.  Both bounds are contract-enforced, so this can only move
        phantom collateral toward phantom books, never a booked balance."""
        return self._send(self.contract.functions.reconcile(
            ss58_to_bytes32(from_hotkey), ss58_to_bytes32(to_hotkey),
            int(amount_rao)))


def client_from_env() -> Optional[CollateralClient]:
    """Read-only client for the live pool (config.py defaults).

    Address/RPC come from FLOW_COLLATERAL_ADDRESS / FLOW_COLLATERAL_RPC
    when set, else the mainnet values pinned in config.py.  Returns
    None if the address is empty or explicitly `off` (f-only weights).
    """
    try:
        from config import FLOW_COLLATERAL_ADDRESS as _addr
        from config import FLOW_COLLATERAL_RPC as _rpc
    except ImportError:
        _addr = os.environ.get("FLOW_COLLATERAL_ADDRESS", "").strip()
        _rpc = os.environ.get("FLOW_COLLATERAL_RPC", MAINNET_RPC)
    addr = (_addr or "").strip()
    if not addr or addr.lower() in ("0", "none", "off", "-"):
        return None
    return CollateralClient(addr, rpc_url=_rpc or MAINNET_RPC)
