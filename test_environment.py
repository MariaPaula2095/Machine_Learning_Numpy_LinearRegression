"""
Tests for the 10x10 environment (Integrante 2).
Run:  python test_environment.py
"""

from collections import deque

from environment import MazeEnvironment10x10, REWARDS

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    assert condition, name


def reachable(env, avoid_danger):
    """BFS from start to target over non-wall (and optionally non-D) cells."""
    seen, queue = {env.start_pos}, deque([env.start_pos])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if not (0 <= n[0] < env.rows and 0 <= n[1] < env.cols) or n in seen:
                continue
            cell = env.grid[n]
            if cell == '#' or (avoid_danger and cell == 'D'):
                continue
            seen.add(n)
            queue.append(n)
    return env.target_pos in seen


def place(env, pos):
    env.reset()
    env.agent_pos = pos


def main():
    env = MazeEnvironment10x10()

    print("== Cell counts ==")
    counts = env.get_cell_counts()
    print(counts)
    check("1 A", counts['A'] == 1)
    check("1 T", counts['T'] == 1)
    check("68 o", counts['o'] == 68)
    check("20 #", counts['#'] == 20)
    check("10 D", counts['D'] == 10)
    check("100 cells", counts['Total'] == 100)
    check("A at (0,0)", env.grid[0, 0] == 'A')
    check("T at (9,9)", env.grid[9, 9] == 'T')

    print("\n== Route exists ==")
    check("route A -> T exists", reachable(env, avoid_danger=False))
    check("safe route A -> T (no D) exists", reachable(env, avoid_danger=True))

    print("\n== Movement ==")
    state = env.reset()
    check("reset returns state 0", state == 0)
    nxt, reward, done, cell = env.step(RIGHT)  # (0,1) is 'o'
    check("valid move Right -> state 1", nxt == 1)
    check("valid move reward = normal step", reward == REWARDS['o'])
    check("valid move cell type 'o', not done", cell == 'o' and not done)
    nxt, _, _, _ = env.step(LEFT)
    check("valid move Left -> back to state 0", nxt == 0)
    nxt, _, _, _ = env.step(DOWN)
    check("valid move Down -> state 10", nxt == 10)
    nxt, _, _, _ = env.step(UP)
    check("valid move Up -> state 0", nxt == 0)

    print("\n== Boundary ==")
    env.reset()
    nxt, reward, done, cell = env.step(UP)
    check("cannot leave grid (Up from corner)", nxt == 0)
    check("boundary penalty", reward == REWARDS['boundary'] and cell == 'boundary')
    nxt, _, _, _ = env.step(LEFT)
    check("cannot leave grid (Left from corner)", nxt == 0)

    print("\n== Wall ==")
    place(env, (0, 2))  # (0,3) is '#'
    nxt, reward, done, cell = env.step(RIGHT)
    check("cannot cross wall", nxt == env.get_state_index((0, 2)))
    check("wall penalty", reward == REWARDS['#'] and cell == '#' and not done)

    print("\n== Danger zone ==")
    place(env, (1, 4))  # (1,5) is 'D'
    nxt, reward, done, cell = env.step(RIGHT)
    check("agent enters D", nxt == env.get_state_index((1, 5)))
    check("danger penalty", reward == REWARDS['D'] and cell == 'D')
    check("episode continues after D", not done)

    print("\n== Target ==")
    place(env, (9, 8 - 1))  # (9,7) -> Right to (9,8) is '#', use (8,9) -> Down
    place(env, (8, 9))
    nxt, reward, done, cell = env.step(DOWN)
    check("agent reaches T", nxt == env.get_state_index(env.target_pos))
    check("goal reward", reward == REWARDS['T'] and cell == 'T')
    check("episode ends at T", done)

    print("\n== Max steps ==")
    env.reset()
    done = False
    for _ in range(env.max_steps):
        _, _, done, _ = env.step(UP)  # bump boundary forever
    check("episode ends at max_steps", done)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
