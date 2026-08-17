// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

/// @title FLOW trade collateral (subnet alpha), live-fed
/// @notice Real skin-in-the-game for FLOW miners on the Bittensor EVM.
///         Positions are scored by the alpha actually at risk behind them
///         (Kelly fraction x this collateral), not by a paper account; every
///         settlement period the alpha lost on losing trades is
///         distributed pro rata to the winners.  Denominated in the
///         subnet's alpha (netuid fixed at deployment, amounts in rao:
///         1 alpha = 1e9 rao).
///
/// Custody model: alpha is not an ERC20; the
/// collateral is *collateral* owned by this contract's mirrored ss58 coldkey, held
/// via the staking-v2 precompile at 0x...0805 and delegated to the miner's
/// own hotkey, so per-hotkey accounting is inherent and the miner cannot
/// move it.  To fund, the caller (whose EVM-mirror coldkey owns the
/// alpha, delegated to `hotkey`) first calls `approve(poolContract,
/// netuid, amount)` on the precompile, then opens the position with
/// `addCollateralSigned` (hotkey-authorized, see below) or tops up an
/// existing one with `addCollateral`.  Only a coldkey can fund by
/// construction: `transferStakeFrom` pulls from the caller's own
/// mirror coldkey.
///
/// One position per hotkey, and only the hotkey can say who runs it:
/// creating a position requires the HOTKEY's own sr25519 signature
/// over (funder, refund coldkey), so nobody can squat a registered
/// hotkey's slot by delegating 1 alpha to it first.  The authorized
/// funder becomes the depositor and the refund coldkey is fixed at
/// that moment; top-ups and withdrawals are depositor-only from then
/// on, so a later hotkey compromise cannot redirect the alpha --
/// the worst a stolen hotkey can do is `evict`, which force-exits a
/// FLAT position back to its own recorded refund coldkey (the
/// miner's, fixed at creation) and frees the slot.  One coldkey can
/// run positions behind any number of hotkeys, each independent.
/// Minimum position is MIN_COLLATERAL (1 alpha); there is no
/// maximum.
///
/// THE MODEL: MINERS POST THEIR OWN BETS, BY HAND, ON CHAIN.  THE
/// OWNER CAN ONLY RESOLVE THEM LATER.
///
///   postTrade / postTradeSigned
///              the miner puts the bet up themselves.  What lands on
///              chain is size and a trade key (regime + id) -- not
///              direction, not levels.  Those live in the encrypted
///              payload on the miner's R2 object and stay under the
///              public timelock; this contract never sees them.  The
///              call reserves the worst case (normally kelly f x
///              collateral) behind that key, either from the
///              depositor EVM key or -- the native path -- with the
///              HOTKEY's sr25519 signature, relayed by any account
///              (the relayer only pays gas; the signature binds every
///              field).  A bet is immutable once posted: no resize,
///              no cancel, because the poster already knows the
///              direction sitting in that payload while the public
///              waits on the timelock.  Expiry derives on-chain from
///              the regime encoded in the key (horizon ceiling + 48h),
///              so a bet cannot be swept out from under its own
///              resolution.  What is never posted can never lose --
///              or win -- a single rao.
///   closeBatch when bets resolve, the owner posts realized losses.
///              A loss is debited immediately: bounded by the bet the
///              miner posted and clamped to the position's balance,
///              it moves from the loser's book into that settlement
///              period's pool.  Wins and flats close with loss 0.
///              The owner cannot open, resize, or extend anything.
///   settle     once per period the owner pays the accumulated pool to
///              the period's winners, pro rata by winning P&L.  The
///              contract enforces the batch zero-sum (payouts == pool)
///              and consumes period ids strictly in order.  A period
///              with no winners rolls its pool into the next period.
///
/// THE WEEK IS CHAIN STATE.  Period k covers [periodZero + (k-1) x
/// PERIOD_SECONDS, periodZero + k x PERIOD_SECONDS), with periodZero
/// fixed forever at deployment.  closeBatch cannot debit into a week
/// that has not started and settle cannot run before a week has
/// ended, so the period circuit breaker (a quarter of a book per
/// period) is a bound per REAL week at any settle velocity, and every
/// boundary is publicly readable (currentPeriod).
///
/// Every owner-posted number is recomputable by anyone once the
/// payloads mature (the public timelock): losses and the weekly split
/// derive from the decrypted payloads, the public tape, and the bets
/// sitting on chain.  A bad post is bounded (a debit can never exceed
/// the miner's own bet or the position, and the pool can only pay
/// held winners) and detectable within the timelock.
///
/// WITHDRAWALS are single-step and instant, gated by margin, not time:
/// a position's open trades reserve their summed exposure, and the
/// depositor can withdraw everything above the reserve at any moment.
/// A trade's worst case is reserved before the trade runs and its loss
/// is debited when it resolves, so no exit shape can outrun a loss.
/// If the owner goes silent, reserves self-release: every open expires
/// by its posted expiry and `sweepExpired` is permissionless, so no
/// owner failure can lock a position.
///
/// There is NO slash on collateral.  The collateral can leave a position through
/// exactly two paths: a settlement debit (zero-sum, to winners) and the
/// depositor's own withdrawal.  The protocol holds no confiscation
/// power over trading capital of any kind.
contract FlowCollateralPool {
    /// staking-v2 precompile (amounts in rao, keys as raw 32-byte pubkeys)
    address public constant ALPHA_PRECOMPILE =
        0x0000000000000000000000000000000000000805;

    /// sr25519 signature verification precompile: hotkeys ARE sr25519
    /// public keys, so the chain itself can check that a message was
    /// signed by the very key a position is booked to
    address public constant SR25519_VERIFY =
        0x0000000000000000000000000000000000000403;

    /// share-pool rounding slack accepted on the pulled amount (rao)
    uint256 public constant AMOUNT_TOLERANCE = 10_000;

    /// minimum position: 1 alpha in rao
    uint256 public constant MIN_COLLATERAL = 1e9;

    /// the longest any reserve can live: expiry derives per bet from
    /// the regime encoded in its trade key (horizon ceiling + 48h
    /// resolution buffer, see _regimeExpiry); this is the D-regime
    /// worst case.  Bounds how long any reserve can survive owner
    /// silence.
    uint256 public constant MAX_OPEN_SECONDS = 336 hours + 48 hours;

    /// max bet, in basis points of the position's balance at post: no
    /// single bet may reserve (and therefore ever move) more than a
    /// quarter of a book.  Mirrors the spec's f cap of 0.25.
    uint256 public constant EXPOSURE_CAP_BPS = 2_500;

    /// one settlement period, ON-CHAIN: period k covers
    /// [periodZero + (k-1) x PERIOD_SECONDS, periodZero + k x
    /// PERIOD_SECONDS).  The week is chain state, not an off-chain
    /// convention: no debit can land in a week that has not started
    /// and no week can settle before it has ended, so the period
    /// circuit breaker means what it says -- at most a quarter of a
    /// book per REAL week, at any settle velocity the owner picks.
    uint256 public constant PERIOD_SECONDS = 168 hours;

    /// minimum physical collateral move attempted during settlement.  The
    /// custody precompile rejects transfers below the runtime's
    /// minimum-collateral-operation threshold, so a below-threshold move
    /// would revert the whole settle.  A settlement move under this is
    /// skipped instead: the rao stays parked on the source hotkey as
    /// booked physical drift, repairable with `reconcile`.  Keep at or
    /// above the runtime minimum.
    uint256 public constant MIN_MOVE = 1e6;   // 0.001 alpha

    struct Position {
        address depositor;      // authorized funder; sole controller
        uint128 balance;        // rao owned; free = balance - openExposure
        uint64 firstFundedAt;
        bytes32 refundColdkey;  // fixed at open under the hotkey's signature
        uint128 openExposure;   // summed worst-case loss of open trades
        uint32 openCount;       // open trades not yet closed or expired
        uint128 pooledOut;      // rao debited from the book but still
                                // physically parked here awaiting settle
    }

    struct Trade {
        uint128 exposure;       // reserved worst-case loss (rao)
        uint64 expiry;          // after this, sweepExpired releases it
    }

    address public owner;
    bytes32 public contractColdkey;  // this contract's ss58 mirror (set once)
    uint256 public immutable netuid;
    /// the challenge genesis: period 1 opens here.  Fixed forever at
    /// deployment, so every period boundary is public chain state.
    uint64 public immutable periodZero;
    uint64 public lastSettledPeriod; // strictly increasing settlement index

    mapping(bytes32 => Position) public positions;
    mapping(bytes32 => mapping(uint64 => Trade)) public trades;
    /// strictly increasing per hotkey: kills replays of hotkey-signed
    /// bet posts (a burned trade key could otherwise be re-posted by a
    /// relayer after its close freed the slot)
    mapping(bytes32 => uint64) public postNonce;
    /// cumulative rao debited from a hotkey inside one settlement
    /// period (the period circuit breaker's ledger)
    mapping(bytes32 => mapping(uint64 => uint128)) public periodDebited;
    mapping(uint64 => uint256) public poolAccum;      // rao per period
    mapping(uint64 => bytes32[]) private poolSrcHk;   // physical sweep list
    mapping(uint64 => uint256[]) private poolSrcAmt;
    /// index+1 of a hotkey's entry in a period's sweep list (0 = none),
    /// so repeated losses from one hotkey aggregate into one source
    /// entry instead of growing the list per losing trade.
    mapping(uint64 => mapping(bytes32 => uint256)) private poolSrcIdx;
    uint256 public totalCollateral;      // sum of position balances (rao)

    event CollateralAdded(bytes32 indexed hotkey, address indexed depositor,
                     uint256 amount, uint256 balance);
    event Evicted(bytes32 indexed hotkey, bytes32 refundColdkey,
                  uint256 paid, uint256 sent);
    event TradePosted(bytes32 indexed hotkey, uint64 indexed tradeKey,
                      uint256 collateralRao, uint64 expiry);
    event TradeClosed(uint64 indexed periodId, bytes32 indexed hotkey,
                      uint64 indexed tradeKey, uint256 lossPosted,
                      uint256 lossCollected);
    event TradeExpired(bytes32 indexed hotkey, uint64 indexed tradeKey,
                       uint256 exposure);
    event PoolRolled(uint64 indexed fromPeriod, uint64 indexed toPeriod,
                     uint256 amount);
    event Settled(uint64 indexed periodId, uint256 pool, uint256 winners);
    event SettleWin(uint64 indexed periodId, bytes32 indexed hotkey,
                    uint256 amount);
    event Withdrawn(bytes32 indexed hotkey, bytes32 refundColdkey,
                    uint256 paid, uint256 sent);
    event Reconciled(bytes32 indexed fromHotkey, bytes32 indexed toHotkey,
                     uint256 amount);
    event SettleMoveSkipped(bytes32 indexed fromHotkey,
                            bytes32 indexed toHotkey, uint256 amount);
    event ContractColdkeySet(bytes32 coldkey);
    event OwnershipTransferred(address indexed from, address indexed to);

    error NotOwner();
    error NotDepositor();
    error NoPosition();
    error BelowMinimum();
    error ZeroAmount();
    error ZeroColdkey();
    error ColdkeyNotSet();
    error ColdkeyAlreadySet();
    error ChainCallFailed();
    error ShortfallReceived();
    error InsufficientFreeCollateral();
    error BadPeriod();
    error LengthMismatch();
    error NotZeroSum();
    error TradeExists();
    error UnknownTrade();
    error NotExpired();
    error BadTradeKey();
    error PeriodNotStarted();
    error PeriodNotOver();
    error ExposureAboveCap();
    error LossExceedsExposure();
    error BadSignature();
    error StaleNonce();
    error NotFlat();
    error ExcessOnly();
    error DeficitOnly();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor(uint256 _netuid, uint64 _periodZero) {
        if (_periodZero == 0) revert ZeroAmount();
        owner = msg.sender;
        netuid = _netuid;
        periodZero = _periodZero;
    }

    /// @notice The settlement week the chain clock is currently in
    ///         (0 before the challenge genesis).
    function currentPeriod() public view returns (uint64) {
        if (block.timestamp < periodZero) return 0;
        return uint64(1 + (block.timestamp - periodZero) / PERIOD_SECONDS);
    }

    // --------------------------------------------------- precompile plumbing
    // State-changing precompile calls must go through a low-level call
    // (interface dispatch does not reach the runtime precompile).

    function _chainCall(bytes memory data) private {
        (bool ok, ) = ALPHA_PRECOMPILE.call(data);
        if (!ok) revert ChainCallFailed();
    }

    function _heldAlpha(bytes32 hotkey) private view returns (uint256) {
        (bool ok, bytes memory out) = ALPHA_PRECOMPILE.staticcall(abi.encodeWithSignature(
            "getStake(bytes32,bytes32,uint256)", hotkey, contractColdkey, netuid));
        if (!ok || out.length < 32) revert ChainCallFailed();
        return abi.decode(out, (uint256));
    }

    function _transferOut(bytes32 destColdkey, bytes32 hotkey, uint256 amount) private {
        _chainCall(abi.encodeWithSignature(
            "transferStake(bytes32,bytes32,uint256,uint256,uint256)",
            destColdkey, hotkey, netuid, netuid, amount));
    }

    /// move contract-owned collateral between hotkeys (owner repair path)
    function _moveAlpha(bytes32 fromHotkey, bytes32 toHotkey, uint256 amount) private {
        if (amount == 0 || fromHotkey == toHotkey) return;
        _chainCall(abi.encodeWithSignature(
            "moveStake(bytes32,bytes32,uint256,uint256,uint256)",
            fromHotkey, toHotkey, netuid, netuid, amount));
    }

    /// @dev Settlement-path collateral move that never reverts the batch.  A
    ///      move below MIN_MOVE, or one the precompile rejects (e.g. a
    ///      runtime dust minimum), is skipped: the rao stays parked on
    ///      `fromHotkey` as booked physical drift, repairable with
    ///      `reconcile`.  Booking is authoritative and already updated by
    ///      the caller, so a skipped physical move cannot lose value or
    ///      break zero-sum -- it only defers the transfer.  This is what
    ///      stops a single dust source from bricking `settle` and locking
    ///      that period (and, via strict ordering, every later one).
    function _moveAlphaSettle(bytes32 fromHotkey, bytes32 toHotkey,
                              uint256 amount) private {
        if (amount == 0 || fromHotkey == toHotkey) return;
        if (amount < MIN_MOVE) {
            emit SettleMoveSkipped(fromHotkey, toHotkey, amount);
            return;
        }
        (bool ok, ) = ALPHA_PRECOMPILE.call(abi.encodeWithSignature(
            "moveStake(bytes32,bytes32,uint256,uint256,uint256)",
            fromHotkey, toHotkey, netuid, netuid, amount));
        if (!ok) emit SettleMoveSkipped(fromHotkey, toHotkey, amount);
    }

    // ------------------------------------------------------------- miner API

    /// @dev The hotkey IS an sr25519 public key; the chain's precompile
    ///      checks that the very key a position is booked to signed
    ///      `digest`.  Shared by every hotkey-authorized entry point.
    function _requireHotkeySig(bytes32 hotkey, bytes32 digest,
                               uint64 nonce, bytes32 r, bytes32 s) private {
        if (nonce <= postNonce[hotkey]) revert StaleNonce();
        (bool ok, bytes memory out) = SR25519_VERIFY.staticcall(
            abi.encodeWithSignature("verify(bytes32,bytes32,bytes32,bytes32)",
                                    digest, hotkey, r, s));
        if (!ok || out.length != 32 || abi.decode(out, (uint256)) != 1)
            revert BadSignature();
        postNonce[hotkey] = nonce;
    }

    /// @dev Pull `amount` from the caller's mirror coldkey onto
    ///      `hotkey` under contract custody and credit the book.
    function _pullIn(bytes32 hotkey, Position storage p, uint256 amount)
        private
    {
        uint256 before = _heldAlpha(hotkey);
        _chainCall(abi.encodeWithSignature(
            "transferStakeFrom(address,address,bytes32,uint256,uint256,uint256)",
            msg.sender, address(this), hotkey, netuid, netuid, amount));
        uint256 received = _heldAlpha(hotkey) - before;
        if (received + AMOUNT_TOLERANCE < amount) revert ShortfallReceived();

        p.balance += uint128(received);
        totalCollateral += received;
        if (p.balance < MIN_COLLATERAL) revert BelowMinimum();
        emit CollateralAdded(hotkey, msg.sender, received, p.balance);
    }

    /// @notice Open `hotkey`'s position -- ONLY the hotkey can say who
    ///         runs it.  The digest binds this chain, this contract,
    ///         the hotkey, the funder (msg.sender, who becomes the
    ///         depositor and whose mirror coldkey the alpha is pulled
    ///         from), the refund coldkey, the amount, and a strictly
    ///         increasing nonce, all under the hotkey's own sr25519
    ///         signature.  Nobody can squat a registered hotkey's slot
    ///         by funding it first: without the signature there is no
    ///         position.  Requires a prior `approve(thisContract,
    ///         netuid, amount)` on the custody precompile from the
    ///         caller.  Collateral is active immediately: it cannot
    ///         retroactively change any trade already open or
    ///         resolved, because exposure is priced per trade at its
    ///         open.
    function addCollateralSigned(bytes32 hotkey, uint256 amount,
                                 bytes32 refundColdkey, uint64 nonce,
                                 bytes32 r, bytes32 s) external {
        if (contractColdkey == bytes32(0)) revert ColdkeyNotSet();
        if (amount == 0) revert ZeroAmount();
        if (refundColdkey == bytes32(0)) revert ZeroColdkey();
        Position storage p = positions[hotkey];
        if (p.depositor != address(0)) revert NotDepositor();
        bytes32 digest = keccak256(abi.encodePacked(
            "FLOWFUND", block.chainid, address(this), hotkey, msg.sender,
            refundColdkey, amount, nonce));
        _requireHotkeySig(hotkey, digest, nonce, r, s);

        p.depositor = msg.sender;
        p.refundColdkey = refundColdkey;
        p.firstFundedAt = uint64(block.timestamp);
        _pullIn(hotkey, p, amount);
    }

    /// @notice Top up an existing position.  Depositor-only;
    ///         `refundColdkey` is ignored (it was fixed, under the
    ///         hotkey's signature, when the position was created).
    ///         Creating a position goes through `addCollateralSigned`.
    function addCollateral(bytes32 hotkey, uint256 amount,
                      bytes32 refundColdkey) external {
        refundColdkey;  // kept for ABI compatibility; top-ups ignore it
        if (contractColdkey == bytes32(0)) revert ColdkeyNotSet();
        if (amount == 0) revert ZeroAmount();
        Position storage p = positions[hotkey];
        if (p.depositor == address(0)) revert NoPosition();
        if (p.depositor != msg.sender) revert NotDepositor();
        _pullIn(hotkey, p, amount);
    }

    /// @notice Force-exit a FLAT position with the hotkey's own
    ///         signature and free the slot.  The payout can go only
    ///         one place: the position's recorded refund coldkey,
    ///         fixed when the position was created -- so this cannot
    ///         redirect a rao, whoever signs.  Two uses: a miner who
    ///         lost the depositor EVM key recovers their alpha to
    ///         their own coldkey and re-adds with a fresh key, and a
    ///         position created against the hotkey owner's wishes can
    ///         be closed out (the funder gets their own alpha back).
    ///         Requires no open trades and no parked pool rao; open
    ///         reserves expire on-chain in <= MAX_OPEN_SECONDS, so an
    ///         evict is never blocked for long.  Anyone may relay.
    function evict(bytes32 hotkey, uint64 nonce, bytes32 r, bytes32 s)
        external
    {
        Position storage p = positions[hotkey];
        if (p.depositor == address(0)) revert NoPosition();
        if (p.openCount != 0 || p.pooledOut != 0) revert NotFlat();
        bytes32 digest = keccak256(abi.encodePacked(
            "FLOWEVICT", block.chainid, address(this), hotkey, nonce));
        _requireHotkeySig(hotkey, digest, nonce, r, s);

        uint256 pay = p.balance;
        bytes32 refund = p.refundColdkey;
        uint256 held = _heldAlpha(hotkey);
        uint256 send = pay > held ? held : pay;  // physical drift clamp
        totalCollateral -= pay;
        delete positions[hotkey];
        if (send > 0) _transferOut(refund, hotkey, send);
        emit Evicted(hotkey, refund, pay, send);
    }

    /// @notice Withdraw `amount` rao immediately, up to the free collateral
    ///         (balance minus the reserve held by open trades).  A
    ///         remainder below MIN_COLLATERAL on a flat, fully swept position
    ///         is paid out too (full exit) and the position is deleted.
    ///         The physical transfer is clamped to the collateral actually
    ///         held on the hotkey minus the rao parked for unsettled
    ///         pools: share-pool rounding can leave a position holding a
    ///         few rao less than its book, and a withdrawal must neither
    ///         brick on a phantom rao nor touch pooled alpha.
    function withdraw(bytes32 hotkey, uint256 amount) external {
        Position storage p = positions[hotkey];
        if (p.depositor == address(0)) revert NoPosition();
        if (p.depositor != msg.sender) revert NotDepositor();
        if (amount == 0) revert ZeroAmount();

        uint256 reserve = p.openExposure;
        uint256 free = p.balance > reserve ? p.balance - reserve : 0;
        if (amount > free) revert InsufficientFreeCollateral();

        uint256 pay = amount;
        bool flat = p.openCount == 0 && p.pooledOut == 0;
        if (flat && p.balance - pay < MIN_COLLATERAL) pay = p.balance;  // full exit

        uint256 held = _heldAlpha(hotkey);
        uint256 parked = p.pooledOut;
        uint256 sendCap = held > parked ? held - parked : 0;
        uint256 send = pay > sendCap ? sendCap : pay;  // physical drift clamp

        bytes32 refund = p.refundColdkey;
        p.balance -= uint128(pay);
        totalCollateral -= pay;
        if (p.balance == 0 && flat) delete positions[hotkey];
        if (send > 0) _transferOut(refund, hotkey, send);
        emit Withdrawn(hotkey, refund, pay, send);
    }

    /// @dev A trade key encodes its regime (key >> 32, 0..3 = A..D).
    ///      Direction and levels stay timelocked, but the regime
    ///      horizon ceilings are protocol constants, so the reserve's
    ///      expiry is derived ON-CHAIN: horizon ceiling plus a 48h
    ///      resolution buffer.  The miner cannot shorten it -- a bet
    ///      cannot be swept out from under its own resolution -- and
    ///      cannot stretch it past the regime's worst case.
    function _regimeExpiry(uint64 tradeKey) private view returns (uint64) {
        uint64 regime = tradeKey >> 32;
        if (regime > 3 || uint32(tradeKey) == 0) revert BadTradeKey();
        uint256 hi = regime == 0 ? 24 hours
                   : regime == 1 ? 48 hours
                   : regime == 2 ? 168 hours
                   : 336 hours;
        return uint64(block.timestamp + hi + 48 hours);
    }

    /// @dev The bet itself: reserve `collateralRao` behind (hotkey, tradeKey)
    ///      until the owner resolves it or it expires.  Immutable once
    ///      posted -- no resize, no cancel -- because the poster
    ///      already knows the direction (it is in the R2 payload, not
    ///      on this chain) while the public waits on the timelock; a
    ///      changeable bet would be a free option.
    function _postTrade(bytes32 hotkey, uint64 tradeKey,
                        uint256 collateralRao) private {
        Position storage p = positions[hotkey];
        if (collateralRao == 0) revert ZeroAmount();
        if (trades[hotkey][tradeKey].expiry != 0) revert TradeExists();
        if (collateralRao * 10_000 > uint256(p.balance) * EXPOSURE_CAP_BPS)
            revert ExposureAboveCap();
        uint64 expiry = _regimeExpiry(tradeKey);
        trades[hotkey][tradeKey] = Trade(uint128(collateralRao), expiry);
        p.openExposure += uint128(collateralRao);
        p.openCount += 1;
        emit TradePosted(hotkey, tradeKey, collateralRao, expiry);
    }

    /// @notice THE MODEL: miners post their own bets, by hand, on
    ///         chain -- the owner can only resolve them later.  Each
    ///         post reserves the bet's worst case (normally kelly f x
    ///         collateral) behind its trade key; the owner cannot
    ///         create, resize, or extend a bet, so nothing can ever be
    ///         at risk that a miner did not personally put there.
    ///         Depositor path; capped at 25% of the book per bet.
    function postTrade(bytes32 hotkey, uint64 tradeKey,
                       uint256 collateralRao) external {
        Position storage p = positions[hotkey];
        if (p.depositor == address(0)) revert NoPosition();
        if (p.depositor != msg.sender) revert NotDepositor();
        _postTrade(hotkey, tradeKey, collateralRao);
    }

    /// @notice Post a bet with the HOTKEY'S OWN SIGNATURE -- the
    ///         native path.  The hotkey a position is booked to IS an
    ///         sr25519 public key, so the chain's precompile verifies
    ///         that the miner's hotkey signed this exact bet.  ANYONE
    ///         may relay the transaction: the sender only pays gas and
    ///         has no authority -- the digest binds chain, contract,
    ///         hotkey, trade key, size and a strictly increasing
    ///         nonce, so a relayer can neither alter a bet nor replay
    ///         one.  Miners need no EVM key of their own to bet.
    function postTradeSigned(bytes32 hotkey, uint64 tradeKey,
                             uint256 collateralRao, uint64 nonce,
                             bytes32 r, bytes32 s) external {
        if (positions[hotkey].depositor == address(0)) revert NoPosition();
        bytes32 digest = keccak256(abi.encodePacked(
            "FLOWPOST", block.chainid, address(this), hotkey, tradeKey,
            collateralRao, nonce));
        _requireHotkeySig(hotkey, digest, nonce, r, s);
        _postTrade(hotkey, tradeKey, collateralRao);
    }

    /// @notice Release the reserve of trades whose expiry has passed
    ///         without a close (owner outage path).  Permissionless: the
    ///         miner can always free their own margin, so no owner
    ///         failure can lock a position.  A loss that was never
    ///         posted before expiry is simply not collected.
    function sweepExpired(bytes32 hotkey, uint64[] calldata tradeKeys)
        external
    {
        Position storage p = positions[hotkey];
        for (uint256 i = 0; i < tradeKeys.length; i++) {
            Trade storage t = trades[hotkey][tradeKeys[i]];
            if (t.expiry == 0) revert UnknownTrade();
            if (block.timestamp <= t.expiry) revert NotExpired();
            uint256 exposure = t.exposure;
            p.openExposure -= uint128(exposure);
            p.openCount -= 1;
            delete trades[hotkey][tradeKeys[i]];
            emit TradeExpired(hotkey, tradeKeys[i], exposure);
        }
    }

    // ------------------------------------------------------------- owner API
    // The owner cannot open anything.  Its entire write surface over
    // trading capital is: resolve bets the miners themselves posted
    // (closeBatch, bounded by each bet), settle the weekly pool
    // (zero-sum enforced), and repair physical rounding drift
    // (reconcile, bounded to phantom collateral).

    /// @notice Post resolved trades.  Losses debit immediately into
    ///         `periodId`'s pool: bounded by the trade's posted exposure
    ///         (hard revert: a close can never take more than the open
    ///         reserved), clamped to the position's balance, and clamped
    ///         by the period circuit breaker (cumulative debits within
    ///         one period never exceed a quarter of the period-start
    ///         balance; the excess is not collected).  Wins
    ///         and flats close with loss 0.  `periodId` must be the
    ///         period currently awaiting settlement or the one after it
    ///         (resolutions straddle the boundary while the previous
    ///         period's last closes are still landing).
    function closeBatch(uint64 periodId,
                        bytes32[] calldata hotkeys,
                        uint64[] calldata tradeKeys,
                        uint256[] calldata lossRao) external onlyOwner {
        if (periodId <= lastSettledPeriod ||
            periodId > lastSettledPeriod + 2) revert BadPeriod();
        // the week is chain state: a debit cannot land in a period
        // that has not started, so the breaker's per-period allowance
        // cannot be drawn early -- at most a quarter of a book can
        // leave per REAL week
        if (periodId > currentPeriod()) revert PeriodNotStarted();
        if (hotkeys.length != tradeKeys.length ||
            hotkeys.length != lossRao.length) revert LengthMismatch();
        for (uint256 i = 0; i < hotkeys.length; i++) {
            bytes32 hk = hotkeys[i];
            Trade storage t = trades[hk][tradeKeys[i]];
            if (t.expiry == 0) revert UnknownTrade();
            if (lossRao[i] > t.exposure) revert LossExceedsExposure();
            Position storage p = positions[hk];
            uint256 loss = lossRao[i];
            if (loss > p.balance) loss = p.balance;  // position cap
            loss = _breakerClamp(hk, periodId, p.balance, loss);
            p.openExposure -= t.exposure;
            p.openCount -= 1;
            delete trades[hk][tradeKeys[i]];
            if (loss > 0) {
                p.balance -= uint128(loss);
                p.pooledOut += uint128(loss);
                totalCollateral -= loss;
                poolAccum[periodId] += loss;
                // aggregate by hotkey: one source entry per losing
                // hotkey per period, not one per losing trade
                uint256 si = poolSrcIdx[periodId][hk];
                if (si == 0) {
                    poolSrcHk[periodId].push(hk);
                    poolSrcAmt[periodId].push(loss);
                    poolSrcIdx[periodId][hk] = poolSrcHk[periodId].length;
                } else {
                    poolSrcAmt[periodId][si - 1] += loss;
                }
            }
            emit TradeClosed(periodId, hk, tradeKeys[i], lossRao[i], loss);
        }
    }

    /// @dev Period circuit breaker: cumulative debits from one miner
    ///      within one settlement period cannot exceed a quarter of its
    ///      period-start balance.  With `d` already debited and `bal`
    ///      the current balance, d + loss <= (bal - loss + d + loss)/4
    ///      rearranges to loss <= (bal - 3d)/4; the excess is simply
    ///      not collected.  Updates the per-period ledger.
    function _breakerClamp(bytes32 hk, uint64 periodId, uint256 bal,
                           uint256 loss) private returns (uint256) {
        uint256 d = periodDebited[hk][periodId];
        uint256 room = bal > 3 * d ? (bal - 3 * d) / 4 : 0;
        if (loss > room) loss = room;
        if (loss > 0) periodDebited[hk][periodId] = uint128(d + loss);
        return loss;
    }

    /// @notice Pay one period's accumulated pool to its winners.
    ///         Attribution is NET per hotkey per period (the spec's
    ///         settlement semantics, flow.compute_collateral_settlement): the
    ///         batch first refunds each book's over-collection -- the
    ///         live feed debits gross per losing trade, so a mixed book
    ///         gets back gross-minus-net -- then the remaining net pool
    ///         pays net winners pro rata by net P&L.  Computed off-chain,
    ///         recomputable by anyone.  Zero-sum enforced: payouts must
    ///         equal the pool exactly, in rao.  Period ids are consumed
    ///         strictly in order, so a batch cannot be replayed.  A
    ///         period with no winners rolls its pool into the next
    ///         period.  Physical collateral parked on the losers moves
    ///         through `winners[0]` as the pivot, each move clamped to
    ///         the collateral actually held.
    function settle(uint64 periodId, bytes32[] calldata winners,
                    uint256[] calldata winRao) external onlyOwner {
        if (periodId != lastSettledPeriod + 1) revert BadPeriod();
        // a week settles only after it has ended, on the chain's own
        // clock: settle velocity is bounded to one real week per
        // period id, whatever the owner does
        if (block.timestamp < uint256(periodZero)
                + uint256(periodId) * PERIOD_SECONDS)
            revert PeriodNotOver();
        if (winners.length != winRao.length) revert LengthMismatch();
        lastSettledPeriod = periodId;

        uint256 pool = poolAccum[periodId];
        bytes32[] storage srcHk = poolSrcHk[periodId];
        uint256[] storage srcAmt = poolSrcAmt[periodId];

        if (winners.length == 0) {
            if (pool > 0) {
                poolAccum[periodId + 1] += pool;
                for (uint256 i = 0; i < srcHk.length; i++) {
                    bytes32 rhk = srcHk[i];
                    uint256 rsi = poolSrcIdx[periodId + 1][rhk];
                    if (rsi == 0) {
                        poolSrcHk[periodId + 1].push(rhk);
                        poolSrcAmt[periodId + 1].push(srcAmt[i]);
                        poolSrcIdx[periodId + 1][rhk] =
                            poolSrcHk[periodId + 1].length;
                    } else {
                        poolSrcAmt[periodId + 1][rsi - 1] += srcAmt[i];
                    }
                }
                emit PoolRolled(periodId, periodId + 1, pool);
            }
            delete poolSrcHk[periodId];
            delete poolSrcAmt[periodId];
            delete poolAccum[periodId];
            emit Settled(periodId, 0, 0);
            return;
        }

        uint256 paid = 0;
        for (uint256 j = 0; j < winners.length; j++) {
            Position storage w = positions[winners[j]];
            if (w.depositor == address(0)) revert NoPosition();
            w.balance += uint128(winRao[j]);
            paid += winRao[j];
            emit SettleWin(periodId, winners[j], winRao[j]);
        }
        if (paid != pool) revert NotZeroSum();
        totalCollateral += pool;

        bytes32 pivot = winners[0];
        for (uint256 i = 0; i < srcHk.length; i++) {
            bytes32 hk = srcHk[i];
            Position storage s = positions[hk];
            uint256 amt = srcAmt[i];
            s.pooledOut -= uint128(amt);
            uint256 held = _heldAlpha(hk);
            if (amt > held) amt = held;  // physical drift clamp; books exact
            _moveAlphaSettle(hk, pivot, amt);
            if (s.balance == 0 && s.openCount == 0 && s.pooledOut == 0 &&
                s.depositor != address(0)) {
                delete positions[hk];
            }
        }
        for (uint256 j = 1; j < winners.length; j++) {
            uint256 amt = winRao[j];
            uint256 held = _heldAlpha(pivot);
            if (amt > held) amt = held;
            _moveAlphaSettle(pivot, winners[j], amt);
        }
        delete poolSrcHk[periodId];
        delete poolSrcAmt[periodId];
        delete poolAccum[periodId];
        emit Settled(periodId, pool, winners.length);
    }

    /// @notice Repair physical drift: move contract-owned collateral from a
    ///         hotkey holding MORE than its booked total (balance plus
    ///         parked pool rao) to one holding LESS.  Both bounds are
    ///         enforced on-chain, so this can only reconcile phantom
    ///         collateral toward phantom books; it cannot touch any booked
    ///         balance, any parked pool, or create value.  Exists
    ///         because runtime share-pool rounding on settlement moves
    ///         can strand a few rao on the wrong hotkey.
    function reconcile(bytes32 fromHotkey, bytes32 toHotkey, uint256 amount)
        external onlyOwner
    {
        if (amount == 0) revert ZeroAmount();
        Position storage f = positions[fromHotkey];
        Position storage t = positions[toHotkey];
        if (_heldAlpha(fromHotkey) <
            uint256(f.balance) + f.pooledOut + amount) revert ExcessOnly();
        if (_heldAlpha(toHotkey) + amount >
            uint256(t.balance) + t.pooledOut) revert DeficitOnly();
        _moveAlpha(fromHotkey, toHotkey, amount);
        emit Reconciled(fromHotkey, toHotkey, amount);
    }

    /// @notice One-shot: record this contract's own ss58 mirror coldkey
    ///         (blake2_256("evm:" ++ address), computed off-chain at
    ///         deployment).  Custody is disabled until set.
    function setContractColdkey(bytes32 coldkey) external onlyOwner {
        if (coldkey == bytes32(0)) revert ZeroColdkey();
        if (contractColdkey != bytes32(0)) revert ColdkeyAlreadySet();
        contractColdkey = coldkey;
        emit ContractColdkeySet(coldkey);
    }

    function transferOwnership(address to) external onlyOwner {
        if (to == address(0)) revert NotOwner();
        emit OwnershipTransferred(owner, to);
        owner = to;
    }

    // ----------------------------------------------------------------- views

    /// @notice Total alpha (rao) booked to `hotkey`.
    function collateralOf(bytes32 hotkey) external view returns (uint256) {
        return positions[hotkey].balance;
    }

    /// @notice The skin the validator weights scores with: the booked
    ///         balance.  Kept under this name so readers of the V1
    ///         interface keep working; there is no pending tier any
    ///         more, exposure is priced per trade at its open.
    function activeCollateralOf(bytes32 hotkey) external view returns (uint256) {
        return positions[hotkey].balance;
    }

    /// @notice Withdrawable right now: balance minus the reserve held by
    ///         open trades (floored at zero).
    function freeCollateralOf(bytes32 hotkey) external view returns (uint256) {
        Position storage p = positions[hotkey];
        uint256 reserve = p.openExposure;
        return p.balance > reserve ? p.balance - reserve : 0;
    }

    /// @notice Distinct loser source entries queued for a period's
    ///         settlement sweep (one per losing hotkey; losses from the
    ///         same hotkey aggregate).  The length `settle` must walk.
    function poolSourceCount(uint64 periodId) external view returns (uint256) {
        return poolSrcHk[periodId].length;
    }

    function tradeInfo(bytes32 hotkey, uint64 tradeKey)
        external view returns (uint256 exposure, uint256 expiry)
    {
        Trade storage t = trades[hotkey][tradeKey];
        return (t.exposure, t.expiry);
    }

    function positionInfo(bytes32 hotkey)
        external
        view
        returns (address depositor, uint256 balance, uint256 firstFundedAt,
                 bytes32 refundColdkey, uint256 openExposure,
                 uint256 openCount, uint256 pooledOut)
    {
        Position storage p = positions[hotkey];
        return (p.depositor, p.balance, p.firstFundedAt, p.refundColdkey,
                p.openExposure, p.openCount, p.pooledOut);
    }
}
