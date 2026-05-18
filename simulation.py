import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


class IteratedPrisonersDilemma:
    COOPERATE = 0
    DEFECT = 1

    def __init__(self, max_rounds=100):
        self.max_rounds = max_rounds
        self.payoff_matrix = {
            (0, 0): (3, 3),
            (0, 1): (0, 5),
            (1, 0): (5, 0),
            (1, 1): (1, 1),
        }
        self.reset()

    def reset(self):
        self.current_round = 0
        self.done = False
        self.history = []
        self.state = ("START", "START")
        return self.state

    def step(self, player_action, opponent_action):
        reward_player, reward_opponent = self.payoff_matrix[
            (player_action, opponent_action)
        ]

        self.current_round += 1
        next_state = (player_action, opponent_action)
        self.state = next_state

        self.history.append({
            "player_action": player_action,
            "opponent_action": opponent_action,
            "reward_player": reward_player,
            "reward_opponent": reward_opponent,
        })

        if self.current_round >= self.max_rounds:
            self.done = True

        return next_state, reward_player, reward_opponent, self.done

    def cooperation_rate(self):
        if not self.history:
            return 0.0

        cooperations = sum(
            1 for record in self.history
            if record["player_action"] == self.COOPERATE
        )

        return cooperations / len(self.history)


class DelayDiscountingTask:
    IMMEDIATE = 0
    DELAYED = 1

    def __init__(self, delay=3, immediate_reward=2, delayed_reward=10, max_steps=50):
        self.delay = delay
        self.immediate_reward = immediate_reward
        self.delayed_reward = delayed_reward
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False
        self.state = "CHOICE"
        return self.state

    def step(self, action, gamma):
        if action == self.IMMEDIATE:
            reward = self.immediate_reward
        elif action == self.DELAYED:
            reward = self.delayed_reward * (gamma ** self.delay)
        else:
            raise ValueError("Invalid action")

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        return self.state, reward, self.done


class TitForTatOpponent:
    def __init__(self):
        self.last_player_action = 0

    def reset(self):
        self.last_player_action = 0

    def choose_action(self):
        return self.last_player_action

    def observe_player_action(self, player_action):
        self.last_player_action = player_action


class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, attention_lapse):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.attention_lapse = attention_lapse
        self.actions = [0, 1]
        self.q_table = {}

    def reset(self):
        pass

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {0: 0.0, 1: 0.0}
        return self.q_table[state][action]

    def choose_action(self, state):
        if random.random() < self.attention_lapse:
            return random.choice(self.actions)

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_0 = self.get_q_value(state, 0)
        q_1 = self.get_q_value(state, 1)

        if q_0 > q_1:
            return 0
        elif q_1 > q_0:
            return 1
        else:
            return random.choice(self.actions)

    def update(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)

        next_best_q = max(
            self.get_q_value(next_state, 0),
            self.get_q_value(next_state, 1)
        )

        new_q = current_q + self.alpha * (
            reward + self.gamma * next_best_q - current_q
        )

        self.q_table[state][action] = new_q


PROFILES = {
    "neurotypical": {
        "alpha": 0.10,
        "gamma": 0.95,
        "epsilon": 0.10,
        "attention_lapse": 0.00,
    },
    "adhd_hyperactive": {
        "alpha": 0.12,
        "gamma": 0.60,
        "epsilon": 0.25,
        "attention_lapse": 0.02,
    },
    "adhd_inattentive": {
        "alpha": 0.08,
        "gamma": 0.85,
        "epsilon": 0.15,
        "attention_lapse": 0.15,
    },
}


def run_ipd_profile(profile_name, profile_params, seed, num_episodes=200, rounds_per_episode=50):
    random.seed(seed)
    agent = QLearningAgent(**profile_params)
    rows = []

    for episode in range(num_episodes):
        env = IteratedPrisonersDilemma(max_rounds=rounds_per_episode)
        opponent = TitForTatOpponent()

        state = env.reset()
        opponent.reset()

        total_reward = 0

        while not env.done:
            player_action = agent.choose_action(state)
            opponent_action = opponent.choose_action()

            next_state, reward_player, reward_opponent, done = env.step(
                player_action,
                opponent_action
            )

            agent.update(state, player_action, reward_player, next_state)
            opponent.observe_player_action(player_action)

            total_reward += reward_player
            state = next_state

        rows.append({
            "task": "ipd",
            "seed": seed,
            "profile": profile_name,
            "episode": episode,
            "reward": total_reward,
            "cooperation_rate": env.cooperation_rate(),
            "delayed_choice_rate": None,
        })

    return rows


def run_delay_profile(profile_name, profile_params, seed, num_episodes=200, steps_per_episode=50):
    random.seed(seed)
    agent = QLearningAgent(**profile_params)
    rows = []

    for episode in range(num_episodes):
        env = DelayDiscountingTask(max_steps=steps_per_episode)

        state = env.reset()
        total_reward = 0
        delayed_choices = 0

        while not env.done:
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action, agent.gamma)
            agent.update(state, action, reward, next_state)

            total_reward += reward

            if action == DelayDiscountingTask.DELAYED:
                delayed_choices += 1

            state = next_state

        rows.append({
            "task": "delay_discounting",
            "seed": seed,
            "profile": profile_name,
            "episode": episode,
            "reward": total_reward,
            "cooperation_rate": None,
            "delayed_choice_rate": delayed_choices / steps_per_episode,
        })

    return rows


def plot_line(df, task, metric, filename, ylabel):
    subset = df[df["task"] == task]

    grouped = (
        subset
        .groupby(["profile", "episode"])[metric]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for profile in grouped["profile"].unique():
        profile_data = grouped[grouped["profile"] == profile]
        plt.plot(
            profile_data["episode"],
            profile_data[metric],
            label=profile
        )

    plt.title(f"{ylabel} over Episodes - {task}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_summary(df):
    summary = (
        df
        .groupby(["task", "profile"])
        .agg(
            mean_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            reward_variance=("reward", "var"),
            mean_cooperation=("cooperation_rate", "mean"),
            cooperation_variance=("cooperation_rate", "var"),
            mean_delayed_choice=("delayed_choice_rate", "mean"),
            delayed_choice_variance=("delayed_choice_rate", "var"),
        )
        .reset_index()
    )

    return summary.round(3)


def format_p_value(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p_value:.3f}"


def cohens_d(group1, group2):
    mean1 = group1.mean()
    mean2 = group2.mean()

    std1 = group1.std(ddof=1)
    std2 = group2.std(ddof=1)

    pooled_sd = ((std1 ** 2 + std2 ** 2) / 2) ** 0.5

    if pooled_sd == 0:
        return 0.0

    return (mean1 - mean2) / pooled_sd


def interpret_effect_size(d_value):
    absolute_d = abs(d_value)

    if absolute_d >= 0.8:
        return "large"
    elif absolute_d >= 0.5:
        return "medium"
    elif absolute_d >= 0.2:
        return "small"
    else:
        return "negligible"


def run_t_tests(df):
    tests = []

    comparisons = [
        ("neurotypical", "adhd_hyperactive"),
        ("neurotypical", "adhd_inattentive"),
        ("adhd_hyperactive", "adhd_inattentive"),
    ]

    test_specs = [
        ("ipd", "reward"),
        ("ipd", "cooperation_rate"),
        ("delay_discounting", "reward"),
        ("delay_discounting", "delayed_choice_rate"),
    ]

    for task, metric in test_specs:
        task_df = df[df["task"] == task]

        for group_a, group_b in comparisons:
            data_a = task_df[task_df["profile"] == group_a][metric].dropna()
            data_b = task_df[task_df["profile"] == group_b][metric].dropna()

            t_stat, p_value = ttest_ind(data_a, data_b, equal_var=False)
            effect_size = cohens_d(data_a, data_b)

            tests.append({
                "task": task,
                "metric": metric,
                "comparison": f"{group_a} vs {group_b}",
                "t_statistic": round(t_stat, 3),
                "p_value_raw": p_value,
                "p_value_reported": format_p_value(p_value),
                "significant_p_0_05": p_value < 0.05,
                "cohens_d": round(abs(effect_size), 3),
                "effect_size_interpretation": interpret_effect_size(effect_size),
            })

    return pd.DataFrame(tests)


if __name__ == "__main__":
    seeds = list(range(20))

    all_rows = []

    for seed in seeds:
        for profile_name, profile_params in PROFILES.items():
            all_rows.extend(
                run_ipd_profile(profile_name, profile_params, seed)
            )
            all_rows.extend(
                run_delay_profile(profile_name, profile_params, seed)
            )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("simulation_results.csv", index=False)

    summary_df = create_summary(results_df)
    summary_df.to_csv("summary_results.csv", index=False)

    ttest_df = run_t_tests(results_df)
    ttest_df.to_csv("ttest_results.csv", index=False)

    plot_line(
        results_df,
        task="ipd",
        metric="reward",
        filename="ipd_reward_curve.png",
        ylabel="Reward"
    )

    plot_line(
        results_df,
        task="ipd",
        metric="cooperation_rate",
        filename="ipd_cooperation_curve.png",
        ylabel="Cooperation Rate"
    )

    plot_line(
        results_df,
        task="delay_discounting",
        metric="reward",
        filename="delay_reward_curve.png",
        ylabel="Reward"
    )

    plot_line(
        results_df,
        task="delay_discounting",
        metric="delayed_choice_rate",
        filename="delay_choice_curve.png",
        ylabel="Delayed Choice Rate"
    )

    print("Simulation complete.")

    print("\nSaved files:")
    print("- simulation_results.csv")
    print("- summary_results.csv")
    print("- ttest_results.csv")
    print("- ipd_reward_curve.png")
    print("- ipd_cooperation_curve.png")
    print("- delay_reward_curve.png")
    print("- delay_choice_curve.png")

    print("\nSummary:")
    print(summary_df)

    print("\nT-test Results:")
    print(ttest_df)