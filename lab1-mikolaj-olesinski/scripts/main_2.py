print(__file__)

from src.utils import Player, Team

team = Team("Team A")

players_data = [
    ("Lewandowski", "forward", 35, 9.2, 25, 8),
    ("Modric", "midfielder", 38, 8.8, 5, 12),
    ("Alaba", "defender", 31, 8.1, 2, 4),
    ("Courtois", "goalkeeper", 31, 9.0, 0, 0),
    ("Vinicius", "forward", 23, 9.1, 18, 10),
]

for name, pos, age, rating, goals, assists in players_data:
    team.add_player(Player(name, pos, age, rating, goals, assists))

# Summary
name, count, avg = team.get_summary()
print(f"Team: {name}, players: {count}, average rating: {avg:.2f}")

# Top scorer
scorer = team.top_scorer()
print(f"Top scorer: {scorer.name} ({scorer.goals} goals)")

# Forwards
forwards = team.find_by_position("forward")
print("Forwards:", [p.name for p in forwards])

# Remove player
team.remove_player("Alaba")
print(f"After removing Alaba: {len(team.players)} players")

print("test")
