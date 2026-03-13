from game import GameConfig, play_game
from engine import Engine

config = GameConfig.small()

DEPTH = 3
FANOUT = 10

eng = Engine(DEPTH, FANOUT, config)

print(f"GAME CONFIG:")
print(f"=================================")
print(f"Num types: {config.num_types}")
print(f"Pool (num goods of each type): ")
for i, count in enumerate(config.init_pool):
    print(f"  Type {i+1}: {count}")
print(f"Goods drawn per round: {config.draw_size}")
print(f"Bundles per draw: {config.num_bundles}")
print(f"Overlap (number of bundles each good appears in): {config.overlap_degree}")
print(f"Rounds: {config.num_rounds}")
print(f"=================================")
print()
print(f"SCORING RULE:")
print(f"=================================")
print(f"""def power_two_score(type_value, count):
    '''Default scoring: 2 and 4 pay double, others pay negative.'''
    if count == 0:
        return 0
    if count == 2:
        return 2 * type_value
    if count == 4:
        return 8 * type_value
    return -type_value * count""")
print(f"=================================")
print()

print(f"Playing game with engine search (depth={DEPTH}, fanout={FANOUT})...")
play_game(config, eng.get_action, verbose=True)
print(f"  Nodes searched: {eng.node_count}, avg {eng.node_count / config.num_rounds:.1f} per move")
