import numpy as np

def create_tiling_grid(low, high, bins=(5, 5), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    low = np.array(low)
    high = np.array(high)
    print("LOW: ", low)
    print("HIGH: ", high)
    tile_sizes = (high - low) / bins

    grid = np.array([np.zeros(bins[0] - 1), np.zeros(bins[1] - 1)])
    for dim in range(0, 2):
        for tile in range(1, bins[dim]):
            grid[dim][tile - 1] = tile * tile_sizes[dim] + low[dim] + offsets[dim]

    return grid


def create_tilings(low, high, bins, offsets):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """

    tilings = []
    for bins_i, offsets_i in zip(bins, offsets):
        tilings.append(create_tiling_grid(low, high, bins_i, offsets_i))
    tilings = tilings

    return tilings


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """

    return tuple(int(np.digitize(sample[dim], grid[dim])) for dim in range(len(sample)))


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """

    encoded = [discretize(sample, tiling) for tiling in tilings]
    if flatten:
        encoded = np.concatenate(encoded)
    return encoded


class FeaturesTable:
    """Simple Q-table."""

    def __init__(self, state_size, action_size):
        """Initialize Q-table.

        Parameters
        ----------
        state_size : tuple
            Number of discrete values along each dimension of state space.
        action_size : int
            Number of discrete actions in action space.
        """
        self.state_size = state_size
        self.action_size = action_size

        # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)

        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("QTable(): size =", self.q_table.shape)


class TiledFeaturesTable:
    """Composite Q-table with an internal tile coding scheme."""

    def __init__(self, low, high, bins, offsets, action_size):
        """Create tilings and initialize internal Q-table(s).

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of state space.
        high : array_like
            Upper bounds for each dimension of state space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
        action_size : int
            Number of discrete actions in action space.
        """
        self.tiling_size = bins[0][0]
        self.layers = len(bins)
        self.tilings = create_tilings(low, high, bins, offsets)
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [FeaturesTable(state_size, self.action_size) for state_size in self.state_sizes]
        print("TiledQTable(): no. of internal tables = ", len(self.q_tables))

    def get_features_vector(self, state, action):
        encoded_state = tile_encode(state, self.tilings)
        vec = np.zeros(self.tiling_size ** 2 * self.layers * self.action_size)
        for i, (x, y) in enumerate(encoded_state):
            vec[action * self.tiling_size ** 2 * self.layers + i * self.tiling_size ** 2 + (
                        x * self.tiling_size + y)] = 1
        return vec

    def get(self, state, action):
        """Get Q-value for given <state, action> pair.

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.

        Returns
        -------
        value : float
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        encoded_state = tile_encode(state, self.tilings)

        q_value = 0.0
        for tiling, q_table in zip(encoded_state, self.q_tables):
            q_value += q_table.q_table[tuple(tiling + (action,))]

        q_value /= len(self.q_tables)

        return q_value

    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value.

        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """

        encoded_state = tile_encode(state, self.tilings)

        for tiling, q_table in zip(encoded_state, self.q_tables):
            old_value = q_table.q_table[tuple(tiling + (action,))]
            q_table.q_table[tuple(tiling + (action,))] = alpha * value + (1.0 - alpha) * old_value
