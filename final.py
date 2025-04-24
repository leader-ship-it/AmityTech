import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import prob1  # used to generate random data
# Load initial delivery order and distance/traffic data

delivery_orders = pd.read_csv("./data/delivery_orders.csv")
d_t = pd.read_csv("./data/distance_traffic_matrix.csv")

# Global variables
reached = 0          # Tracks if all orders were completed and returned to depot
discount = 1         # Time step discount to simulate delay penalty
visited = []         # Tracks visited locations to discourage revisits


# Calculates travel time from a given location to all others
def calculate_time(src, x):
    if src == 0:  # Special handling for depot
        for i, j, k, l in zip(d_t["from_location_id"], d_t["to_location_id"],
                              d_t["base_time_min"], d_t["traffic_multiplier"]):
            if i == "DEPOT":
                x[int(j[-3:].lstrip('0'))] = float(k) * float(l)
            else:
                break
    else:
        for i, j, k, l in zip(d_t["from_location_id"], d_t["to_location_id"],
                              d_t["base_time_min"], d_t["traffic_multiplier"]):
            # Skip depot rows
            if j != "DEPOT" and i != "DEPOT" and isinstance(i, str) and int(i[-3:].lstrip('0')) == src:
                x[int(j[-3:].lstrip('0'))] = float(k) * float(l)
            elif (not isinstance(i, str)) or (i != "DEPOT" and int(i[-3:].lstrip('0')) == src + 1):
                break
            elif j == "DEPOT" and i != "DEPOT" and int(i[-3:].lstrip('0')) == src:
                x[0] = float(k) * float(l)
    return x

# Populates order list with highest priority values for each location
def calculate_priority(x):
    for i, j in zip(delivery_orders["priority"], delivery_orders["delivery_location_id"]):
        if x[int(j[-3:].lstrip('0'))] < int(i):
            x[int(j[-3:].lstrip('0'))] = int(i)
    return x


# Generates randomized orders and traffic conditions
def randomize():
    global d
    d = prob1.generate_delivery_orders(80, prob1.generate_ids("LOC", 40))
    return d
class delivery(gym.Env):
    def __init__(self):
        super(delivery, self).__init__()

        # 41 possible locations (including depot at index 0)
        self.action_space = spaces.Discrete(41)
        self.locations = 41
        self.current_location = 0

        # Observation space includes:
        # - current location
        # - vector of order priorities at each location
        # - travel times from current location
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + self.locations + self.locations,),  # Flattened size: 1 for current_location, +locations for orders, +locations for travel_times
            dtype=np.float32
        )

        self.steps = 0
        self.orders_list = np.zeros(self.locations, dtype=np.int8)
    def _get_obs(self):
        return np.concatenate((
            [self.current_location / self.locations],  # Normalize current location
            self.orders_list.copy() / 3.0,                         # Normalize order priorities
            self.travel_times.copy() / 113.5       # Normalize travel times
        )).astype(np.float32)

    def reset(self, seed=None, options=None):
            global delivery_orders, d_t, reached, discount, visited
            super().reset(seed=seed)

            # Reset globals
            reached = 0
            discount = 0
            visited = []

            # Environment state reset
            self.steps = 0
            self.current_location = 0

            # Regenerate new order list and traffic conditions
            a= randomize()
            delivery_orders = a
            # Update travel time and order list
            self.travel_times = np.zeros(41)
            self.travel_times = calculate_time(0, self.travel_times)
            self.orders_list = np.zeros(41, dtype="int8")
            self.orders_list = calculate_priority(self.orders_list)
            obs=self._get_obs()
            return obs, {}
    def _get_reward(self, delivered, priority, travel_time):
        if delivered:
            # Give more weight to priority, reduce penalty for time
            priority_weight = 3.0 * (priority / 3.0)  # Stronger reward for high priority
            time_penalty = (travel_time / 113.5)
            return priority_weight - time_penalty # Max reward ~2.0
        else:
            # Constant step penalty + time penalty when NOT delivering
            return - (travel_time / 113.5) * 2.0  # Max penalty = -2.0

    def step(self, action):
        global reached, discount, visited
        terminated = False
        truncated = False
        self.steps += 1 # increase steps
        self.current_location = action
        
        if self.orders_list[action]>0:
            reward=self._get_reward(True,self.orders_list[action],self.travel_times[action])
            self.orders_list[action]=0
        else:
            reward=self._get_reward(False,self.orders_list[action],self.travel_times[action])-0.08
        # Update travel times based on new location
        self.travel_times = np.zeros(41)
        self.travel_times = calculate_time(action, self.travel_times)
        # End episode after too many steps
        if self.steps > 50:

            truncated = True
        # If all orders completed
        elif np.all(self.orders_list == 0):
            reward+=10
            terminated=True
            
        obs=self._get_obs()
        
        
        return obs, reward, terminated, truncated, {}
