if self.sharks and self.fishes:
            combinations = it.product(self.sharks, self.fishes)
            for shark, fish in combinations:
                if self.collision_space.check_collision(shark, fish):
                    n_companion_shark = 0
                    for companion_shark in self.sharks:
                        if(companion_shark != shark and self.collision_space.check_collision(companion_shark, fish)):
                            self._on_shark_fish_collision(companion_shark, fish)
                            n_companion_shark += 1
                    if n_companion_shark > 0:
                        self._on_shark_fish_collision(shark, fish)
                    else:
                        self.collision_space.perform_collision(shark, fish)
                else:
                    fish.survived_n_steps += 1