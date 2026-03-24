print(__file__)


results = {
    "Real Madrid": 3,
    "Barcelona": 1,
    "Atletico": 2,
    "Sevilla": 0,
}

scored = {team for team, goals in results.items() if goals > 0}
print("Teams that scored:", scored)

ratings = [6.5, 7.0, 8.5, 5.0, 9.0, 7.5]
for i in range(len(ratings)):
    r = ratings[i]
    if r >= 8.5:
        label = "legendary"
    elif r >= 7.0:
        label = "good"
    else:
        label = "bad"

    print(f"Player {i + 1}: {r} — {label}")


match = ("Real Madrid", "Barcelona", 3, 1)
home, away, home_goals, away_goals = match

print(f"Match: {home} vs {away} — {home_goals}:{away_goals}")
