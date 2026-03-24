from dataclasses import dataclass


@dataclass
class Player:
    name: str
    position: str
    age: int
    avg_rating: float
    goals: int
    assists: int


class Team:
    def __init__(self, name: str):
        self.name = name
        self.players: list = []

    def add_player(self, player: Player):
        self.players.append(player)

    def remove_player(self, name: str):
        self.players = [p for p in self.players if p.name != name]

    def find_by_position(self, position: str) -> list:
        return [p for p in self.players if p.position == position]

    def top_scorer(self) -> Player:
        if not self.players:
            return None
        return max(self.players, key=lambda p: p.goals)

    def average_rating(self) -> float:
        if not self.players:
            return 0.0
        return sum(p.avg_rating for p in self.players) / len(self.players)

    def get_summary(self) -> tuple:
        return (self.name, len(self.players), self.average_rating())
