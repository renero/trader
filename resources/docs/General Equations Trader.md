**General Equations Trader**

The price can be the **buy price** or the **price** at a given time. The Buy price = $p_{0}$. The price at $t$ is $p_{t}$. And the number of shares is $n$.

The **initial cost** ($k$) of the position is given by the expression:

$$
k = n \cdot p_{0}
$$

The **value** ($v_{t}$) of the position(s) at any time is given by the amount of shares and the price of the option at that time:

$$
v_{t} = p_{t} \cdot n
$$

Each position can be played *bullish* or *bearish*. That will be indicated by the superscript $ \square^{B} $ or $ \square^{b} $. These superscript only make sense when computing the profit obtained by the position, or its performance, if computed in absolute terms.

The **performance** is the ratio between price evolution and buy price (bullish or bearish) at a given time. If the price increase in a bullish position, the performance will be how much that price increases with respect to the acquisition price. In a bearish position, the difference is taken reversing the order of factors, as it is suposed to decrease.

$$
r^{B}_{t} = \frac{p_{t} - p_{0}}{p_{0}} ;\ \ r^{b}_{t} = \frac{p_{0} - p_{t}}{p_{0}}
$$

From the performance, we can easily compute the **profit** ($M_{t}$) obtained by a given position, by simply multiplying its performance by its value.

$$
M_{t} = r_{t} \cdot v_{t}
$$

**Sell Operations**

When selling partially or totally a position, we sell a number of options from it: $n_s$. From the sell operation, we're interested in knowing what is the **cost** and the **profit** that the sell operation generates.

The **cost** is the amount of money that we spent when acquiring a given package of shares. We compute it by multiplying the amount of shares sold by the acquisition price price.

$$
k_{s} = n_s \cdot p_0
$$

And the benefit from that operaion is given by the product between the performance and the cost, to obtain the portion of that amount of money that corresponds to the price difference:

$$
M_{s, t} = r_t \cdot k_{s} = r_{t} \cdot n_s \cdot p_0
$$

