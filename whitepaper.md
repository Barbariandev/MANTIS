# MANTIS: The Ultimate Signal Machine


MANTIS is a Bittensor subnet running on Netuid 123 designed to evaluate and reward any information that can contribute to the prediction of Bitcoin price movements. Miners upload timelocked 100-dimensional embedding vectors to an R2 bucket, submit the R2 bucket public dev URL to the chain, validators fetch this once every block, along with the current BTC prices, after config.LAG blocks miner submissions are detimelocked, BTC hourly returns are computed, and every TASK_INTERVAL saliences are computed, and weights are set. The orchestrator also publishes miner historical data to an R2 bucket for other validators to fetch.


The core idea is to reward miners proportional to the information-theoretic value they add to predicting future price changes. By doing so, MANTIS turns the problem of price prediction into a collaborative, competitive game: many agents propose signals, and the network automatically learns which combinations of signals best explain the price movements, rewarding those agents accordingly. This paper provides a technical overview of how the MANTIS subnet functions, detailing the commit-reveal submission process, the ΔMAE based scoring mechanism for miner contributions, and how the design leverages counterfactual analysis to handle signal overlap and synergy. We further explore the emergent dynamics that arise from this mechanism.

MANTIS Subnet Architecture and Workflow
Miner Embeddings & Time-Lock Commitments: In MANTIS, each miner is expected to submit a fixed-size feature vector (embedding) of length 100 at regular intervals. Each item in the vector must be between -1 and 1.

. This 100-dimensional embedding encodes the miner’s information or prediction about the Bitcoin market for the upcoming interval. To ensure fairness and prevent miners from copying others after seeing their submissions, MANTIS validators expect the embeddings to be submitted timelocked.

Miners publish their R2 bucket public development URL to the chain, the name of the file containing the HEX of the timelocked preds is expected to be their hotkey ss58.


Every block, a cycle runs:
The validator fetches each miner’s latest committed embedding from the distributed storage and appends it to that miner’s historical data record along with the corresponding Bitcoin price at that time. 

This LAG parameter defines how far ahead in the future the subnet is trying to predict – in this case, 300 blocks, which corresponds to the price change horizon the model will learn to forecast. By continuously accumulating a dataset of (embedding history, price history), the validator prepares the ground for periodic training/evaluation. Evaluation Phase (ΔMAE Scoring): Every set number of blocks (the TASK_INTERVAL, e.g. 360 blocks

The validator regenerates weights. The validator first ensures that a sufficient history is present (at least as long as the LAG, so that a prediction model can be trained)

It then decrypts all the time-locked embeddings collected (other than those not expected to be decryptable due to having been submitted within the last LAG blocks)

With all miners’ plaintext embeddings for past blocks accessible, the validator constructs a training dataset: the inputs are the miners’ embeddings over a series of past time steps, and the target output is the subsequent Bitcoin price movement (percentage change after   LAG blocks). Losses to be used for scoring are collected LAG indices ahead of indices being trained on. So after collecting a loss for indice 3000 backprop is run on block 2700.

The choice of learning target can be a regression (predicting the exact price change or return) or classification (predicting direction of movement), but in the current design it uses lagged percentage price changes as the prediction target
github.com

A multi-layer perceptron (MLP) model is then trained on this dataset to predict the Bitcoin price change from the combined miner signals

This model serves as a baseline for measuring feature importance. Once a baseline model is trained on all available features (all miners’ embeddings), the validator evaluates each miner’s marginal contribution using a counterfactual removal approach: for each miner $i$, the validator zeroes out (masks) that miner’s feature set in the dataset, trains a new model and measures the degradation in the model’s performance

. The performance metric used is Mean Absolute Percentage Error (MAE). The increase in error when miner $i$’s data is removed – i.e. $\Delta \text{MAE}i = \text{MAE}{\text{without } i} - \text{MAE}_{\text{all signals}}$ – represents the unique predictive value of miner $i$’s signal. Higher $\Delta$MAE means the miner was providing information that significantly helped the prediction (since without it the error rises)

. This ΔMAE scoring system thus yields a raw representation of each miners informational value. Miners whose embeddings were largely redundant or uninformative will see little to no change in error when removed (ΔMAE ≈ 0), whereas miners whose embeddings carried novel  predictive signals will incur a substantial error increase when omitted (large positive ΔMAE). The validator then normalizes these scores into weights summing to 1. Then weights are set.




Counterfactual Modeling: Resolving Redundancy and Leveraging Synergy

A key innovation in MANTIS is the use of counterfactual modeling.
Below, we examine why this approach is superior to naive individual scoring:

Baseline: Individual Scoring (Naive Approach). In a simpler scheme , one would evaluate each miner’s embedding on its own by training a separate model per miner or simply having miners submit predictions.

For example, a validator might compute the predictive accuracy (or error) of a single-miner model using only miner $i$’s embedding as input, and then rank miners by these individual accuracies. While straightforward, this approach fails to account for information overlap: if two miners submit very similar signals, both models might show good individual accuracy, leading the system to reward them both highly even though they are largely redundant to each other.
This is a large problem on existing subnets tackling similar problems such as SN8 and SN55.

Conversely, a miner whose embeddings or signals are not individually predictive may be undervalued, as there is the possibility that signal carries predictive power when combined with others (such as with funding rate and open interest along with many other feature pairs, historically). In other words, additive individual evaluation cannot detect higher-order interactions, and is therefore unsuitable for incentivizing the highest quality NETWORK outputs. 

Counterfactual Group Evaluation (MANTIS Approach). MANTIS instead evaluates miners collectively by training one model on all signals together and then measuring each miner’s marginal contribution via a counterfactual removal. This approach inherently addresses the drawbacks above:

If 2 or more miners are providing the same or highly correlated signals weights for all UIDs will drop dramatically, as removing any given miner will not drop the performance of the network significantly, none will receive much weight. This leads to scenarios where if 2 or more miners have the same strategy it is in their best interests to coordinate to only run 1 UID. This cooperative exploitation becomes much harder as the N_UIDs providing the redundant information increases, additionally if either ceases to provide the signals it becomes profitable to provide them again for the other. Therefore at no time is it in miners best interests to not contribute their signals.


I am hopeful that in the coming weeks MANTIS will be able to produce highly predictive models. When this is the case the timelock scheme will be changed to accomodate the use of signals immediately.

Once we accumulate a decently sized database of historical embeddings we will be experimenting to find the most performant model architectures for the task.  The task may also shift somewhat, though at no point should it shift enough to invalidate previously useful embeddings.

MANTIS is a network of 256 UIDs each scrambling to produce the most novel, predictive, and easily interpretable information possible. The Ultimate Signal Machine.
